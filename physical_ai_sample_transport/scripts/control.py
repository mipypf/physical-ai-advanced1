from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parents[1]
TRAYS = np.array([[.025,.090,.024],[.065,.080,.024],[.100,.060,.024]])

class Lab:
    def __init__(self):
        self.model = m = mujoco.MjModel.from_xml_path(str(ROOT/'assets/lab.xml'))
        self.data = d = mujoco.MjData(m)
        self.plan = mujoco.MjData(m)
        self.site = m.site('grasp').id
        self.tool = m.body('crane_plus_link4').id
        self.adr = np.array([m.joint(f'crane_plus_joint{j}').qposadr[0] for j in range(1,5)])
        self.sample_adr = [m.joint(f'sample_joint_{i}').qposadr[0] for i in range(3)]
        d.qpos[self.adr] = [0,0,0,0]
        d.ctrl[:4] = d.qpos[self.adr]
        d.ctrl[4] = -.35
        mujoco.mj_forward(m,d)
        self.maxheight = np.zeros(3)
        self.maxspeed = 0.
        self.joint_violation = 0.

    def position(self, i):
        return self.data.xpos[self.model.body(f'sample_{i}').id].copy()

    def ik(self, xyz):
        m,p = self.model,self.plan
        ranges = m.jnt_range[:4]
        def residual(q):
            p.qpos[self.adr]=q
            mujoco.mj_forward(m,p)
            theta=np.pi-min(.65,max(0,float(xyz[2])-.025)*10)
            yaw=np.arctan2(xyz[1],xyz[0]);axis=np.array([np.sin(theta)*np.cos(yaw),np.sin(theta)*np.sin(yaw),np.cos(theta)])
            return np.r_[p.site_xpos[self.site]-xyz, .06*(p.site_xmat[self.site].reshape(3,3)[:,2]-axis)]
        starts = [self.data.qpos[self.adr].copy(),np.array([np.arctan2(xyz[1],xyz[0]),.5,-1.,-1.6])]
        results=[least_squares(residual,np.clip(q,ranges[:,0]+.001,ranges[:,1]-.001),bounds=(ranges[:,0]+.0001,ranges[:,1]-.0001),max_nfev=100,xtol=1e-9,ftol=1e-9,gtol=1e-9) for q in starts]
        best=min(results,key=lambda r: np.linalg.norm(r.fun))
        if np.linalg.norm(best.fun[:3])>.003:
            raise RuntimeError(f'IK unreachable: {xyz}, residual={best.fun}')
        return best.x

    def step(self):
        mujoco.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all(): raise RuntimeError('Nonfinite state')
        for i in range(3): self.maxheight[i]=max(self.maxheight[i], self.position(i)[2])
        self.maxspeed = max(self.maxspeed,float(np.max(np.abs(self.data.qvel[:4]))))
        q=self.data.qpos[self.adr]; r=self.model.jnt_range[:4]
        self.joint_violation=max(self.joint_violation,float(max(0,np.max(r[:,0]-q),np.max(q-r[:,1]))))

    def latch(self,i):
        m,d=self.model,self.data
        distance=np.linalg.norm(d.site_xpos[self.site]-self.position(i))
        if distance>.008:
            print('LATCH DEBUG', 'q',d.qpos[:5], 'ctrl',d.ctrl, 'site',d.site_xpos[self.site], 'sample',self.position(i))
            import PIL.Image
            renderer=mujoco.Renderer(m,height=720,width=1280);renderer.update_scene(d,camera='overview');PIL.Image.fromarray(renderer.render()).save(ROOT/'artifacts/failure.png');renderer.close()
            print('CONTACTS',[(m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist)) for c in d.contact])
            raise RuntimeError(f'Latch too far: {distance:.4f}m')
        # Equality uses the CURRENT relative pose, never the initial scene pose.
        b=m.body(f'sample_{i}').id
        relative=d.xmat[self.tool].reshape(3,3).T@(d.xpos[b]-d.xpos[self.tool])
        inverse=np.empty(4); mujoco.mju_negQuat(inverse,d.xquat[self.tool])
        quat=np.empty(4); mujoco.mju_mulQuat(quat,inverse,d.xquat[b])
        m.eq_data[i,3:6]=relative; m.eq_data[i,6:10]=quat
        d.eq_active[i]=True
        return float(distance)

    def release(self,i):
        self.data.eq_active[i]=False
