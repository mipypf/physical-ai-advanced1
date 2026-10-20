"""Physical transport baseline. The gripper uses an explicitly disclosed ideal latch."""
import argparse,json
import numpy as np
from control import Lab,TRAYS,ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',action='store_true');ap.add_argument('--ai',action='store_true');ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    lab=Lab();m,d=lab.model,lab.data
    rng=np.random.default_rng(args.seed)
    # IDs are shuffled across pickup locations; a position-based routing rule cannot solve this.
    labels=rng.permutation(3)
    colors=np.array([[.85,.15,.12,1],[.12,.65,.25,1],[.15,.35,.9,1]])
    for i,label in enumerate(labels):
        b=m.body(f'sample_{i}').id
        m.geom_rgba[m.geom_bodyid==b]=colors[label]
    writer=renderer=None
    if args.video:
        import mujoco,imageio.v2 as imageio
        from PIL import Image,ImageDraw,ImageFont
        renderer=mujoco.Renderer(m,height=540,width=960)
        writer=imageio.get_writer(str(ROOT/'artifacts/baseline_demo.mp4'),fps=15,codec='libx264',quality=7,macro_block_size=None)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    steps=0; phase='Initialize'; predictions=[]; results=[]
    def tick(seconds):
        nonlocal steps
        for _ in range(round(seconds/m.opt.timestep)):
            lab.step();steps+=1
            if writer is not None and steps%33==0:
                renderer.update_scene(d,camera='overview');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;im=Image.fromarray(renderer.render()).resize((1280,720));draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,1280,98),fill=(16,23,33));draw.text((22,12),'MATERIALS LAB | CRANE+ V2 | sample transport',font=font,fill='white')
                draw.text((22,48),phase,font=font,fill=(110,210,255))
                draw.rectangle((0,650,1280,720),fill=(16,23,33));draw.text((22,658),'Prototype: ideal latch grasp | known pickup positions | simulation',font=font,fill=(255,210,110))
                draw.text((22,687),'CNN identifies ID markers; marker color does not imply material properties.',font=font,fill='white')
                writer.append_data(np.asarray(im))
    def move(target,seconds=2.5,hand=None):
        nonlocal phase
        start=d.ctrl[:4].copy();goal=lab.ik(np.array(target))
        if hand is not None:d.ctrl[4]=hand
        n=round(seconds/m.opt.timestep)
        for k in range(n):
            t=(k+1)/n;s=t*t*(3-2*t);d.ctrl[:4]=start+(goal-start)*s
            tick(m.opt.timestep)
        tick(.3)
    tick(2)
    if args.ai:
        from vision import classify_scene
        predictions,confidences=classify_scene(lab)
    else:
        predictions=list(map(int,labels));confidences=[1.]*3
    print('Labels:',labels,'Predictions:',predictions,flush=True)
    for i,label in enumerate(predictions):
        if args.ai and confidences[i]<.8: raise RuntimeError('Low confidence: stop')
        start=lab.position(i);dest=TRAYS[label]
        phase=f'ID {"ABC"[label]} -> analysis station {label+1} | approach'
        move([start[0],start[1],.090])
        phase=f'ID {"ABC"[label]} | lower and latch'
        move(start)
        gap=lab.latch(i);d.ctrl[4]=.15;tick(.5)
        phase=f'ID {"ABC"[label]} | lift and transport'
        move([start[0],start[1],.090])
        lifted=lab.position(i)[2]
        move([dest[0],dest[1],.090])
        phase=f'ID {"ABC"[label]} | lower and release'
        move(dest)
        lab.release(i);d.ctrl[4]=-.35;tick(.7)
        phase=f'ID {"ABC"[label]} | retract'
        move([dest[0],dest[1],.090])
        tick(.5)
        actual=lab.position(i);error=float(np.linalg.norm(actual[:2]-dest[:2]))
        r={'sample':i,'true_id':int(labels[i]),'predicted_id':int(label),'confidence':float(confidences[i]),'latch_gap_m':gap,'lifted_center_z_m':float(lifted),'final_position_m':actual.tolist(),'target_m':dest.tolist(),'xy_error_m':error,'success':bool(labels[i]==label and error<.015 and lifted>.045 and abs(actual[2]-.024)<.009 and np.linalg.norm(d.qvel[m.joint(f'sample_joint_{i}').dofadr[0]:m.joint(f'sample_joint_{i}').dofadr[0]+6])<.1)}
        results.append(r);print(json.dumps(r),flush=True)
    phase='Transport complete | verifying destination trays';tick(5)
    for r in results:
        actual=lab.position(r['sample']);dest=TRAYS[r['predicted_id']]
        r['final_position_m']=actual.tolist();r['xy_error_m']=float(np.linalg.norm(actual[:2]-dest[:2]))
        r['success']=bool(r['success'] and r['xy_error_m']<.015 and abs(actual[2]-.024)<.009)
    phase='Transport complete'
    report={'seed':args.seed,'ai_enabled':args.ai,'grasp':'ideal proximity-gated weld; not friction-only grasp','localization':'known simulator positions','robot_sample_contacts':'disabled in this ideal-latch baseline; sample-bench/tray contacts enabled','results':results,'all_success':all(r['success'] for r in results) and lab.joint_violation<.001 and lab.maxspeed<2.,'max_joint_speed_rad_s':lab.maxspeed,'max_joint_limit_violation_rad':lab.joint_violation,'simulation_seconds':float(d.time)}
    (ROOT/f'artifacts/result_seed_{args.seed}.json').write_text(json.dumps(report,indent=2))
    if writer:writer.close();renderer.close()
    print(json.dumps(report,indent=2),flush=True)
    if not report['all_success']:raise SystemExit(1)

if __name__=='__main__':main()
