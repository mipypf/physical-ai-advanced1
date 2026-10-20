"""Convert upstream CRANE+ V2 xacro to MuJoCo, then add laboratory scene.
Original robot geometry, joints, limits, and inertias come from rt-net/crane_plus.
"""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import xacro
import mujoco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
SRC = ROOT / 'vendor/crane_plus/crane_plus_description'


def build():
    OUT.mkdir(exist_ok=True)
    # Reuse the upstream constants and macro, without ROS-specific plugins.
    original = (SRC / 'urdf/crane_plus.urdf.xacro').read_text()
    tree = ET.fromstring(original)
    ns = '{http://ros.org/wiki/xacro}'
    ET.register_namespace('xacro', 'http://ros.org/wiki/xacro')
    wrapper = ET.Element('robot', {'name': 'crane_plus'})
    ET.SubElement(wrapper, 'xacro:include', {'filename': str(SRC / 'urdf/crane_plus.xacro')})
    for e in tree:
        if e.tag in [ns+'property', 'material']:
            wrapper.append(e)
    ET.SubElement(wrapper, 'link', {'name': 'base_link'})
    macro = ET.SubElement(wrapper, 'xacro:crane_plus', {'parent': 'base_link'})
    ET.SubElement(macro, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    text = ET.tostring(wrapper, encoding='unicode').replace('ns0:', 'xacro:').replace('xmlns:ns0=', 'xmlns:xacro=')
    # ElementTree generates its own namespace declaration; normalize it.
    text = text.replace('xmlns:xacro="http://ros.org/wiki/xacro" xmlns:xacro="http://ros.org/wiki/xacro"', 'xmlns:xacro="http://ros.org/wiki/xacro"')
    (OUT/'wrapper.xacro').write_text(text)
    urdf = ET.fromstring(xacro.process_file(str(OUT/'wrapper.xacro')).toxml())
    for mesh in urdf.findall('.//mesh'):
        relative = mesh.get('filename').replace('package://crane_plus_description/', '')
        path = SRC / relative
        name = path.parent.name + '_' + path.name
        shutil.copy2(path, OUT/name)
        mesh.set('filename', str(OUT/name))
    ext = ET.SubElement(urdf, 'mujoco')
    ET.SubElement(ext, 'compiler', {'discardvisual': 'false', 'fusestatic': 'false', 'balanceinertia': 'true'})
    ET.ElementTree(urdf).write(OUT/'crane_plus.urdf')
    model = mujoco.MjModel.from_xml_path(str(OUT/'crane_plus.urdf'))
    mujoco.mj_saveLastXML(str(OUT/'robot.xml'), model)
    scene = ET.parse(OUT/'robot.xml').getroot()
    scene.set('model','CRANE+ V2 sample transport laboratory')
    scene.find('compiler').set('meshdir', str(OUT))
    ET.SubElement(scene, 'option', {'timestep':'0.002', 'integrator':'implicitfast', 'gravity':'0 0 -9.81'})
    visual = ET.SubElement(scene,'visual')
    ET.SubElement(visual,'global',{'offwidth':'1280','offheight':'720'})
    ET.SubElement(visual,'headlight',{'ambient':'.45 .45 .45', 'diffuse':'.6 .6 .6'})
    world = scene.find('worldbody')
    ET.SubElement(world, 'light', {'pos':'0 -.3 1', 'dir':'0 0 -1', 'diffuse':'.8 .8 .8'})
    ET.SubElement(world,'geom', {'name':'bench','type':'plane','size':'.65 .5 .02','rgba':'.16 .20 .25 1','friction':'1 .01 .001','conaffinity':'6'})
    import numpy as np
    eye=np.array([.39,-.48,.39]);focus=np.array([.055,.01,.135]);z=(eye-focus)/np.linalg.norm(eye-focus);x=np.cross([0,0,1],z);x=x/np.linalg.norm(x);y=np.cross(z,x)
    ET.SubElement(world, 'camera', {'name':'overview','pos':' '.join(map(str,eye)),'xyaxes':' '.join(map(str,np.r_[x,y])),'fovy':'38'})
    ET.SubElement(world, 'camera', {'name':'top','pos':'.12 0 .8','xyaxes':'1 0 0 0 1 0'})
    # Robot visuals are preserved, collision meshes use their convex hulls in MuJoCo.
    for joint in scene.findall('.//joint'):
        if joint.get('name'):
            joint.set('damping','.04')
            joint.set('armature','.002')
            joint.set('frictionloss','.01')
    for geom in scene.findall('.//geom'):
        if geom.get('type') == 'mesh' and geom.get('contype') != '0':
            geom.set('contype','4');geom.set('conaffinity','1');geom.set('group','3')
    for body in scene.findall('.//body'):
        if body.get('name','').startswith('crane_plus_'):
            body.set('gravcomp','1')
    link4 = scene.find('.//body[@name="crane_plus_link4"]')
    ET.SubElement(link4,'site',{'name':'grasp','pos':'0 0 .121','size':'.003','rgba':'1 .7 0 1'})
    # Baseline: use a declared ideal latch after a proximity check. This is not a contact grasp benchmark.
    equality = ET.SubElement(scene,'equality')
    colors = ['.85 .15 .12 1','.12 .65 .25 1','.15 .35 .9 1']
    positions = [(.075,-.045),(.105,-.010),(.075,.025)]
    for i, ((x,y),color) in enumerate(zip(positions,colors)):
        body = ET.SubElement(world,'body',{'name':f'sample_{i}','pos':f'{x} {y} .018'})
        ET.SubElement(body,'freejoint',{'name':f'sample_joint_{i}'})
        ET.SubElement(body,'geom',{'name':f'sample_geom_{i}','type':'cylinder','size':'.011 .018','mass':'.015','rgba':color,'friction':'1 .02 .002','condim':'4','contype':'2','conaffinity':'3'})
        ET.SubElement(body,'geom',{'type':'cylinder','pos':'0 0 .020','size':'.012 .003','mass':'.002','rgba':color,'contype':'2','conaffinity':'3'})
        ET.SubElement(equality,'weld',{'name':f'latch_{i}','body1':'crane_plus_link4','body2':f'sample_{i}','active':'false','solref':'.006 1'})
        tx,ty = [(.025,.090),(.065,.080),(.100,.060)][i]
        ET.SubElement(world,'geom',{'name':f'tray_{i}','type':'box','pos':f'{tx} {ty} .003','size':'.018 .019 .003','rgba':'.30 .35 .40 1','conaffinity':'6'})
    actuator = ET.SubElement(scene,'actuator')
    for j in range(1,5):
        name=f'crane_plus_joint{j}'
        ET.SubElement(actuator,'position',{'name':f'motor{j}','joint':name,'kp':'20','kv':'1','forcerange':'-1.5 1.5'})
    ET.SubElement(actuator,'position',{'name':'hand','joint':'crane_plus_joint_hand','kp':'5','kv':'.1','forcerange':'-.4 .4'})
    ET.indent(scene)
    ET.ElementTree(scene).write(OUT/'lab.xml')
    test = mujoco.MjModel.from_xml_path(str(OUT/'lab.xml'))
    print(f'Built CRANE+ scene: {test.nq} qpos, {test.nu} actuators, {test.nbody} bodies')

if __name__ == '__main__':
    build()
