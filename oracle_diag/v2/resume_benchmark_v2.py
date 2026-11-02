"""Resume the immutable supplemental benchmark without repeating completed members."""
from support import *
import shutil

def main():
    original=HERE/'benchmark_new_tasks.py'
    old=OUT/'stage_b_new_tasks'
    dest=OUT/'resume1/stage_b_new_tasks'
    source=original.read_text(encoding='utf-8')
    changes=[
        ("dest=OUT/'stage_b_new_tasks';fresh(dest);torch.set_num_threads(2);results={}",
         "dest=OUT/'resume1/stage_b_new_tasks';fresh(dest);torch.set_num_threads(2);results={};reuse=OUT/'stage_b_new_tasks'"),
        ("        tar,full=load_target_buffer(env,0);agent=build_bellman_agent(env,0,tar);dim=tar.state.shape[1]",
         """        if base=='walker2d':
            for task in ['walker2d_gravity','walker2d_morph']:
                for ext in ['.json','_errors.npz']:
                    srcfile=reuse/(task+ext);assert srcfile.is_file()
                    shutil.copy2(str(srcfile),str(dest/srcfile.name))
                    assert sha(srcfile)==sha(dest/srcfile.name)
                result=read(reuse/(task+'.json'));a=np.load(reuse/(task+'_errors.npz'))
                assert all(np.isfinite(a[k]).all() for k in a.files)
                results[task]=result
            continue
        tar,full=load_target_buffer(env,0);agent=build_bellman_agent(env,0,tar);dim=tar.state.shape[1]"""),
        ("            for seed in range(5):\n                torch.manual_seed(seed);m=network(x.shape[1],dim+1);opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed)",
         """            saved=np.load(reuse/'hopper_split.npz')
            regenerated=dict(train=train,raw_indices=rawids,episode=episode,ymean=ym,ystd=ys,xmean=mean,xstd=std)
            for key,value in regenerated.items():assert np.array_equal(saved[key],value),'Split/normalization changed: '+key
            for seed in range(5):
                if seed<4:
                    model_file=reuse/('hopper_model_%d.pt'%seed);metadata=reuse/('hopper_member_%d.json'%seed)
                    state=torch.load(str(model_file),map_location='cpu')
                    assert all(torch.isfinite(t).all() for t in state.values())
                    m=network(x.shape[1],dim+1);m.load_state_dict(state);m.eval();models.append(m)
                    for file in [model_file,metadata]:
                        shutil.copy2(str(file),str(dest/file.name));assert sha(file)==sha(dest/file.name)
                    print('REUSED HOPPER MODEL',seed,flush=True)
                    continue
                torch.manual_seed(seed);m=network(x.shape[1],dim+1);opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed)""")]
    for before,after in changes:
        assert source.count(before)==1,'Resume transform ambiguous'
        source=source.replace(before,after)
    namespace=dict(__name__='resumed_benchmark',__file__=str(original),shutil=shutil)
    exec(compile(source,str(original)+' [resume wrapper]','exec'),namespace)
    namespace['main']()
    write(dest/'resume_provenance.json',dict(original_script=str(original),original_script_sha256=sha(original),
        reused_walker_tasks=['walker2d_gravity','walker2d_morph'],reused_hopper_members=[0,1,2,3],
        restarted_hopper_members=[4],source_directory=str(old),split_and_normalization_identical=True))
if __name__=='__main__':main()
