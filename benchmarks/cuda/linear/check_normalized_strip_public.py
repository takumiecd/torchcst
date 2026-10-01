"""Installed-src integration: independent small truth; optional full-step timing.

Large benchmark is NOT an independent all-A oracle. No experiments imports.
Run from checkout or install torchcst and point --tests-file at the oracle helper.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import runpy
import os
import sys
wheel_root=os.environ.get("CST_PUBLIC_PACKAGE_ROOT") or os.environ.get("CST_PUBLIC_WHEEL_ROOT")
if wheel_root:sys.path.insert(0,wheel_root)
import torch
from torchcst.nn import NormalizedStripLinear


def check(a,b,tol=3e-4):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a.double(),b.double(),atol=tol,rtol=tol)
    return {'max':float((a.double()-b.double()).abs().max()),
            'rel_l2':float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))}


def small_gate(helper):
    sizes=(1024,4,4);p=helper['mixed'](torch.float32,'cuda')
    p[0,2]=511.25;p[1,2]=512.5;p[2:4,2]=512.02978515625
    generator=torch.Generator(device='cuda').manual_seed(21)
    x=torch.randn(2,3,16,device='cuda',generator=generator)
    dy=torch.randn(2,3,1024,device='cuda',generator=generator)
    reports={}
    for memory in ('full','window'):
        model=NormalizedStripLinear(helper['chart'](sizes,dtype=torch.float32,device='cuda'),p,memory=memory)
        xx=x.clone().requires_grad_();actual=model(xx)
        tp=model.p.detach().double().requires_grad_();tx=x.double().requires_grad_()
        truth=tx@helper['oracle'](tp,sizes,stored_dtype=torch.float32).T
        ag=torch.autograd.grad(actual,(xx,model.p),dy)
        tg=torch.autograd.grad(truth,(tx,tp),dy.double())
        reports[memory]={'y':check(actual,truth),'dx':check(ag[0],tg[0]),'dp':check(ag[1],tg[1])}
        # Capture entire training step, then compare multiple replays from the
        # exact post-capture P/m/v/step state, with a current-centre mutation.
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,fused=True,capturable=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        def step(m,o):
            o.zero_grad(set_to_none=True);loss=(m(x)*dy).sum()/dy.numel();loss.backward();o.step()
        with torch.cuda.stream(stream):
            for _ in range(3):step(model,opt)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):step(model,opt)
        torch.cuda.synchronize()
        eager=NormalizedStripLinear(helper['chart'](sizes,dtype=torch.float32,device='cuda'),model.p.detach(),memory=memory)
        eo=torch.optim.AdamW(eager.parameters(),lr=1e-4,weight_decay=.01,fused=True,capturable=True)
        eo.load_state_dict(copy.deepcopy(opt.state_dict()))
        with torch.no_grad():
            # Route/support changes across the row-window boundary after capture.
            model.p[0,2].add_(1.);eager.p[0,2].add_(1.)
            model.p[1,1].fill_(math.log(.199));eager.p[1,1].fill_(math.log(.199))
        for _ in range(3):
            graph.replay();step(eager,eo);torch.cuda.synchronize()
            check(model.p,eager.p);check(model.p.grad,eager.p.grad)
            for key in ('exp_avg','exp_avg_sq','step'):check(opt.state[model.p][key],eo.state[eager.p][key])
        # Updated independent, unscaled all-five check, not just optimizer parity.
        xx=x.clone().requires_grad_();y=model(xx);tp=model.p.detach().double().requires_grad_()
        tx=x.double().requires_grad_();truth=tx@helper['oracle'](tp,sizes,stored_dtype=torch.float32).T
        aa=torch.autograd.grad(y,(xx,model.p),dy);bb=torch.autograd.grad(truth,(tx,tp),dy.double())
        reports[memory]['updated']={'y':check(y,truth),'dx':check(aa[0],bb[0]),'dp':check(aa[1],bb[1])}
    return reports


def benchmark(helper,n,profile,memory,dense=False):
    h,j=(32,32) if n==1024 else (64,128);sizes=(n,h,j)
    origin=(-(n-1)/2,-(h-1)/4,-(j-1)/4);a=round(.05*n*n)
    gen=torch.Generator(device='cpu').manual_seed(21)
    p=torch.empty(a,5);p[:,0]=torch.rand(a,generator=gen)-.5
    p[:,1]=math.log(3. if profile=='broad' else .199)
    p[:,2:]=torch.rand(a,3,generator=gen)*torch.tensor([(count-1)*s for count,s in zip(sizes,(1.,.5,.5))])+torch.tensor(origin)
    if profile=='sharp':
        for axis,(spacing,o) in enumerate(zip((1.,.5,.5),origin)):
            u=(p[:,axis+2]-o)/spacing;near=torch.floor(u+.5)
            p[:,axis+2]=o+near*spacing+(u-near)*.04
    sha=hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest()
    p=p.cuda();torch.manual_seed(21)
    x=torch.randn(128,n,device='cuda',requires_grad=True)
    target=torch.randn(128,n,device='cuda')
    if dense:
        model=torch.nn.Linear(n,n,bias=False,device='cuda');model.weight.data.uniform_(-.01,.01)
    else:model=NormalizedStripLinear(helper['chart'](sizes,origin,dtype=torch.float32,device='cuda'),p,memory=memory)
    del p
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,fused=True,capturable=True)
    def step():
        opt.zero_grad(set_to_none=True);x.grad=None
        loss=(model(x)*target).sum()/(128*n);loss.backward();opt.step()
    for _ in range(3):step()
    def measured(call):
        samples=[]
        for _ in range(7):
            torch.cuda.synchronize();begin=time.perf_counter()
            call();torch.cuda.synchronize()
            samples.append((time.perf_counter()-begin)*1000)
        return {'median_ms':sorted(samples)[3],'samples_ms':samples,
                'clock':'perf_counter with CUDA synchronization; no event instrumentation'}
    eager=measured(step)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):step()
    ms=measured(graph.replay);torch.cuda.synchronize()
    return {'n':n,'a':a,'m':128,'profile':profile,'memory':memory,'dense':dense,
        'initial_p_sha256':sha,'eager_ms':eager['median_ms'],'graph_ms':ms['median_ms'],'eager_wall':eager,'graph_wall':ms,
        'allocated_before_capture':baseline,'max_allocated_capture_replay':torch.cuda.max_memory_allocated(),
        'max_reserved_capture_replay':torch.cuda.max_memory_reserved(),
        'optimizer_steps':float(next(iter(opt.state.values()))['step']),
        'scope':'full-shape timing, no independent all-A oracle'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--tests-file',default='tests/test_normalized_strip_public.py')
    ap.add_argument('--bench',action='store_true');ap.add_argument('--sizes',nargs='+',type=int,default=[1024,8192])
    ap.add_argument('--profiles',nargs='+',default=['broad','sharp']);ap.add_argument('--memory',nargs='+',default=['full','window'])
    ap.add_argument('--dense',action='store_true');ap.add_argument('--skip-small',action='store_true')
    args=ap.parse_args();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!='NVIDIA L4':raise RuntimeError('this protocol requires NVIDIA L4')
    helper=runpy.run_path(args.tests_file)
    import torchcst
    module_path=str(Path(torchcst.__file__).resolve())
    if wheel_root and not Path(module_path).is_relative_to(Path(wheel_root).resolve()):
        raise RuntimeError('torchcst was not imported from requested extracted wheel')
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'tf32':False,'torchcst_file':module_path}
    if not args.skip_small:
        result['small_independent']=small_gate(helper)
        gc.collect();torch.cuda.empty_cache()
        Path(args.output).write_text(json.dumps(result,indent=2))
    if args.bench:
        result['benchmarks']=[]
        for n in args.sizes:
            if n not in (1024,8192):raise ValueError('sizes must be1024 or8192')
            for profile in args.profiles:
                if profile not in ('broad','sharp'):raise ValueError('unknown profile')
                for memory in args.memory:
                    result['benchmarks'].append(benchmark(helper,n,profile,memory))
                    gc.collect();torch.cuda.empty_cache()
                    Path(args.output).write_text(json.dumps(result,indent=2))
            if args.dense:
                result['benchmarks'].append(benchmark(helper,n,'broad','full',True))
                gc.collect();torch.cuda.empty_cache()
                Path(args.output).write_text(json.dumps(result,indent=2))
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
