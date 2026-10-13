"""Single-record encoder inference; see ../MODEL_LOADING.md."""
import argparse
from pathlib import Path
import numpy as np
import torch
from released_models import load_teacher,load_stmem,load_vitmae,prepare_stmem,prepare_vitmae,encode_vitmae


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['teacher','stmem','vitmae'],required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--geometry')
    ap.add_argument('--stmem-repo')
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    x=np.load(args.input,allow_pickle=False)
    if args.model=='teacher':
        if not args.geometry:
            ap.error('--geometry is required for the full geometry-aware teacher')
        if x.shape not in [(15,224,224),(16,224,224)]:
            raise ValueError('Expected (15,224,224) or (16,224,224) CMR array')
        if not np.isfinite(x).all() or x.min()<0 or x.max()>1:
            raise ValueError('CMR must be pre-normalized to [0,1]')
        g=np.load(args.geometry,allow_pickle=False)['token_xyz'][:15]
        if g.shape!=(15,16,3) or not np.isfinite(g).all():
            raise ValueError('Expected finite token_xyz (15,16,3)')
        model=load_teacher(args.checkpoint,args.device)
        with torch.no_grad():
            _,z=model(torch.from_numpy(x[:15].astype(np.float32)).unsqueeze(0).to(args.device),
                      geom=torch.from_numpy(g.astype(np.float32)).unsqueeze(0).to(args.device))
    elif args.model=='stmem':
        if not args.stmem_repo:
            ap.error('--stmem-repo is required')
        model=load_stmem(args.checkpoint,args.stmem_repo,args.device)
        t=torch.from_numpy(prepare_stmem(x,args.stmem_repo)).unsqueeze(0).to(args.device)
        with torch.no_grad(): z=model(t)
    else:
        model=load_vitmae(args.checkpoint,args.device)
        t=torch.from_numpy(prepare_vitmae(x)).unsqueeze(0).to(args.device)
        z=encode_vitmae(model,t)
    if not torch.isfinite(z).all():
        raise ValueError('Non-finite encoder output')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.save(output,z.float().cpu().numpy())
    print(f'Saved {output}: {tuple(z.shape)}')


if __name__=='__main__': main()
