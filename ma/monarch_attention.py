from enum import StrEnum

import torch
import torch.nn as nn

from ma.ma_torch import monarch_attention_torch
from ma.ma_triton import monarch_attention_triton

Tensor = torch.Tensor


class PadType(StrEnum):
    pre = "pre"
    post = "post"


class MonarchAttention(nn.Module):

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        pad_type: PadType,
        impl: str | None = None,
        dtype: torch.dtype | None = None,
        use_cuda_graph: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type

        # Auto-detect implementation: use Triton on CUDA, torch otherwise
        if impl is None:
            impl = "triton" if torch.cuda.is_available() else "torch"

        self.impl = impl
        self.dtype = dtype  # None = use input dtype, or specify torch.float16/bfloat16/float32
        self.use_cuda_graph = use_cuda_graph and impl == "triton"

        # CUDA graph caching (keyed by input shape and dtype)
        self._cuda_graphs: dict[tuple, tuple] = {}  # shape -> (graph, q_buf, k_buf, v_buf, out_buf)

    def _get_graph_key(self, query: Tensor) -> tuple:
        """Get cache key for CUDA graph based on input shape and dtype."""
        return (query.shape, query.dtype, query.device)

    def _run_with_cuda_graph(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Run forward pass using CUDA graphs for reduced kernel launch overhead.

        Note: CUDA graphs provide ~30% speedup by eliminating kernel launch overhead,
        but require input/output buffer copies. The net benefit depends on input size.
        For maximum performance, the returned tensor should be consumed before the
        next forward call, as the output buffer is reused.
        """
        key_tuple = self._get_graph_key(query)

        if key_tuple not in self._cuda_graphs:
            # Create static buffers for graph capture
            q_buf = query.clone()
            k_buf = key.clone()
            v_buf = value.clone()

            # Capture the graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out_buf = monarch_attention_triton(
                        q_buf,
                        k_buf,
                        v_buf,
                        None,  # CUDA graphs don't support dynamic masks
                        self.num_steps,
                        self.block_size,
                        self.pad_type == PadType.pre,
                    )
            torch.cuda.current_stream().wait_stream(s)

            self._cuda_graphs[key_tuple] = (g, q_buf, k_buf, v_buf, out_buf)

        # Replay the graph with new inputs
        g, q_buf, k_buf, v_buf, out_buf = self._cuda_graphs[key_tuple]
        q_buf.copy_(query)
        k_buf.copy_(key)
        v_buf.copy_(value)
        g.replay()

        # Note: We don't clone out_buf to save memory copy. The caller should
        # consume the output before the next forward call.
        return out_buf

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Convert to specified dtype if set
        original_dtype = query.dtype
        if self.dtype is not None and query.dtype != self.dtype:
            query = query.to(self.dtype)
            key = key.to(self.dtype)
            value = value.to(self.dtype)

        # Use CUDA graphs if enabled and no attention mask
        if self.use_cuda_graph and attention_mask is None:
            output = self._run_with_cuda_graph(query, key, value)
        else:
            output = (
                monarch_attention_triton
                if self.impl == "triton"
                else monarch_attention_torch
            )(
                query,
                key,
                value,
                attention_mask,
                self.num_steps,
                self.block_size,
                self.pad_type == PadType.pre,
            )

        # Convert back to original dtype if we changed it
        if self.dtype is not None and original_dtype != self.dtype:
            output = output.to(original_dtype)

        return output

    def get_matrix(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        value = torch.eye(seq_len, device=query.device).expand(
            batch_size, num_heads, seq_len, seq_len
        )
        return self.forward(query, key, value, attention_mask)
