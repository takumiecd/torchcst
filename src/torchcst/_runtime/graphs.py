"""Reusable CUDA graphs with explicit changing inputs and isolated outputs."""

from threading import RLock

import torch


class CapturedCall:
    def __init__(self, function):
        self.function = function
        self.entries = {}
        self.lock = RLock()

    def __call__(self, *args):
        if args[0].device.type != "cuda":
            return self.function(*args)
        key = tuple((a.device, a.dtype, a.shape, a.stride()) for a in args)
        with self.lock, torch.cuda.device(args[0].device), torch.no_grad():
            if key not in self.entries:
                static = tuple(a.clone() for a in args)
                stream = torch.cuda.Stream(device=args[0].device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    # Compilation and allocation are outside the captured region.
                    for _ in range(2):
                        self.function(*static)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = self.function(*static)
                self.entries[key] = (static, graph, outputs, torch.cuda.Event())
                self.entries[key][3].record(stream)
            static, graph, outputs, event = self.entries[key]
            torch.cuda.current_stream().wait_event(event)
            for target, source in zip(static, args):
                target.copy_(source)
            graph.replay()
            result = tuple(x.clone() for x in outputs)
            event.record()
            return result
