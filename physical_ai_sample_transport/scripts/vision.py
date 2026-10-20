"""Small CNN classifies visible carrier ID markers in top-camera RGB crops.
Pickup coordinates are supplied by the simulator in this baseline, not inferred.
"""
import os,sys
if "--train-only" in sys.argv:
    os.environ["MUJOCO_GL"]="disable"
import json
import numpy as np
import mujoco
import torch
from torch import nn
from PIL import Image
from control import Lab,ROOT

COLORS=np.array([[.85,.15,.12,1],[.12,.65,.25,1],[.15,.35,.9,1]])

class MarkerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(24,3))
    def forward(self,x):return self.net(x)


def crops(lab,renderer):
    renderer.update_scene(lab.data,camera='top')
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    image=renderer.render()
    cam=lab.model.camera('top').id
    height,width=image.shape[:2];focal=height/(2*np.tan(np.deg2rad(lab.model.cam_fovy[cam])/2))
    result=[]
    for i in range(3):
        p=lab.position(i)+[0,0,.023]
        local=lab.data.cam_xmat[cam].reshape(3,3).T@(p-lab.data.cam_xpos[cam])
        u=width/2+focal*local[0]/(-local[2]);v=height/2-focal*local[1]/(-local[2])
        # Crop a fixed metric neighborhood including marker and immediate background.
        radius=int(round(.020*focal/(-local[2])))
        patch=image[round(v)-radius:round(v)+radius,round(u)-radius:round(u)+radius]
        if not patch.size:raise RuntimeError('Empty camera crop')
        result.append(np.asarray(Image.fromarray(patch).resize((32,32))).copy())
    return np.stack(result)


def generate_dataset(seed,scenes):
    rng=np.random.default_rng(seed);lab=Lab()
    renderer=mujoco.Renderer(lab.model,height=384,width=384)
    images=[];labels=[]
    for j in range(scenes):
        if j%10==0: print(f"render seed={seed} scene={j}/{scenes}",flush=True)
        ids=rng.permutation(3)
        for i,label in enumerate(ids):
            b=lab.model.body(f'sample_{i}').id
            color=COLORS[label].copy();color[:3]=np.clip(color[:3]*rng.uniform(.65,1.20,3),.03,1.)
            lab.model.geom_rgba[lab.model.geom_bodyid==b]=color
            adr=lab.sample_adr[i]
            lab.data.qpos[adr:adr+2]=lab.model.qpos0[adr:adr+2]+rng.uniform(-.004,.004,2)
        lab.model.light_diffuse[:]=rng.uniform(.45,.95)
        lab.model.light_ambient[:]=rng.uniform(.05,.25)
        mujoco.mj_forward(lab.model,lab.data)
        patches=crops(lab,renderer)
        for p,label in zip(patches,ids):
            # Sensor noise and exposure perturbation; split is by independent rendered scene.
            for _ in range(8):
                a=p.astype(np.float32)*rng.uniform(.80,1.15)+rng.normal(0,3,p.shape)
                images.append(np.clip(a,0,255).astype(np.uint8));labels.append(int(label))
    renderer.close()
    return np.stack(images),np.array(labels)


def tensor(images,device):return torch.tensor(images.transpose(0,3,1,2).copy(),device=device,dtype=torch.float32)/255.


def train():
    torch.manual_seed(2026);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Generating independently randomized train/test render scenes',flush=True)
    dataset=np.load(ROOT/'artifacts/vision_dataset.npz');x,y,tx,ty=[dataset[k] for k in ['x','y','tx','ty']]
    Image.fromarray(np.concatenate(list(x[::max(1,len(x)//9)][:9]),axis=1)).save(ROOT/'artifacts/training_examples.png')
    a,b=tensor(x,device),torch.tensor(y,device=device);ta,tb=tensor(tx,device),torch.tensor(ty,device=device)
    net=MarkerCNN().to(device);opt=torch.optim.Adam(net.parameters(),lr=.003)
    for epoch in range(35):
        net.train();order=torch.randperm(len(a),device=device)
        for indices in order.split(96):
            opt.zero_grad();loss=nn.functional.cross_entropy(net(a[indices]),b[indices]);loss.backward();opt.step()
        if epoch%10==0:print('epoch',epoch,'loss',float(loss.detach()),flush=True)
    net.eval()
    with torch.no_grad():pred=net(ta).argmax(1);acc=float((pred==tb).float().mean())
    confusion=torch.zeros(3,3,dtype=torch.int64)
    for truth,p in zip(tb.cpu(),pred.cpu()):confusion[truth,p]+=1
    report={'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'train_render_scenes':30,'test_render_scenes':12,'train_images':len(x),'test_images':len(tx),'test_accuracy':acc,'confusion_matrix':confusion.tolist(),'scope':'synthetic marker ID classification; simulator-provided crop locations; no material-property inference or real-world validation'}
    torch.save(net.cpu().state_dict(),ROOT/'artifacts/marker_cnn.pt')
    (ROOT/'artifacts/vision_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if acc<.95:raise SystemExit('Insufficient synthetic validation accuracy')


def classify_scene(lab):
    # Keep OSMesa rendering and CUDA training in separate processes.
    # This tiny classifier runs inference on CPU; GPU is used for training.
    device='cpu';net=MarkerCNN().to(device)
    net.load_state_dict(torch.load(ROOT/'artifacts/marker_cnn.pt',weights_only=True,map_location=device));net.eval()
    renderer=mujoco.Renderer(lab.model,height=384,width=384)
    patches=crops(lab,renderer);renderer.close()
    Image.fromarray(np.concatenate(list(patches),axis=1)).resize((576,192)).save(ROOT/'artifacts/inference_crops.png')
    with torch.no_grad():scores=net(tensor(patches,device)).softmax(1);confidence,pred=scores.max(1)
    return pred.cpu().tolist(),confidence.cpu().tolist()

if __name__=='__main__':
    import sys,subprocess
    if '--dataset-only' in sys.argv:
        x,y=generate_dataset(100,30);tx,ty=generate_dataset(200,12)
        np.savez_compressed(ROOT/'artifacts/vision_dataset.npz',x=x,y=y,tx=tx,ty=ty)
        print('Dataset saved',flush=True)
    elif '--train-only' in sys.argv: train()
    else:
        subprocess.run([sys.executable,'-X','faulthandler',__file__,'--dataset-only'],check=True)
        subprocess.run([sys.executable,'-X','faulthandler',__file__,'--train-only'],check=True)

