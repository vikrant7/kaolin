# Copyright (c) 2019, NEVADA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kaolin.graphics import DIBRenderer as Renderer
from kaolin.graphics.dib_renderer.utils.mesh import face2pfmtx
from kaolin.rep import TriangleMesh
import argparse
import imageio
import numpy as np
import os
import torch
import tqdm
#from torchvision.models import resnet18

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
#torch.cuda.set_device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
###########################
# Settings
###########################

CAMERA_DISTANCE = 1.5#2
CAMERA_ELEVATION = 30
MESH_SIZE = 1
HEIGHT = 600#256
WIDTH = 600#256


def parse_arguments():
    parser = argparse.ArgumentParser(description='Kaolin DIB-R Example')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'banana.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')

    return parser.parse_args()


def main():
    args = parse_arguments()

    ###########################
    # Load mesh
    ###########################

    mesh = TriangleMesh.from_obj(args.mesh, with_vt=True, texture_res=5)
    vertices = mesh.vertices	
    faces = mesh.faces.long()
    face_textures = mesh.face_textures
    textures = mesh.textures
    uvs = mesh.uvs
    
    ## trying to corect for pytorch coordinates
    #uvs[:, 1] = 1 - uvs[:, 1]
    #uvs = uvs * 2 - 1
    pfmtx = face2pfmtx(faces.numpy())
    # Expand such that batch size = 1

    vertices      = vertices[None, :, :].to(device)
    faces         = faces[None, :, :].to(device)
    face_textures = face_textures[None, :, :].to(device)
    textures      = textures[None, :, :, :].to(device)
    uvs           = uvs[None, :, :].to(device)

    textures = textures.permute([0, 3, 1, 2])
    #vertices = vertices.unsqueeze(0)
    print(vertices.shape, faces.shape, textures.shape, uvs.shape)
    ###########################
    # Normalize mesh position
    ###########################

    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = (vertices - vertices_middle) * MESH_SIZE

    ###########################
    # Generate vertex color
    ###########################

    #vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
    #vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
    #colors = (vertices - vert_min) / (vert_max - vert_min)

    ###########################
    #  Mat, Light, and Shine  #
    ###########################
    bs = 1
    material = np.array([[0.1, 0.1, 0.1],
                         [1.0, 1.0, 1.0],
                         [0.4, 0.4, 0.4]], dtype=np.float32).reshape(-1, 3, 3)
    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)# this was 100
    tfmat = torch.from_numpy(material).repeat(bs, 1, 1).to(device)
    tfshi = torch.from_numpy(shininess).repeat(bs, 1).to(device)

    lightdirect = np.array([0, 1, 0], dtype=np.float32).reshape((bs, 3))#2 * np.random.rand(bs, 3).astype(np.float32) - 1
    #lightdirect[:, 2] += 2
    tflight = torch.from_numpy(lightdirect)
    tflight_bx3 = tflight.to(device)
    
    ###########################
    # Render
    ###########################
   
    renderer = Renderer(HEIGHT, WIDTH, device, mode='Phong')
    #renderer.renderer.set_smooth(pfmtx)
    
    loop = tqdm.tqdm(list(range(0, 180, 4)))
    loop.set_description('Drawing')

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.output_path, 'example.gif'), mode='I')
    
    theta1=np.round(np.random.uniform(0,360,20),2)
    theta2=tqdm.tqdm(list(np.round(np.random.uniform(0,360,20),2)))
    
    #net = resnet18(pretrained=True).cuda()
    for azimuth in loop:
        renderer.set_look_at_parameters([90 - azimuth],
                                        [CAMERA_ELEVATION],
                                        [CAMERA_DISTANCE])
        predictions, _, _ = renderer(points=[vertices, faces[0].long()], uv_bxpx2=uvs, ft_fx3=face_textures[0], 
                                         texture_bx3xthxtw=textures, lightdirect_bx3=tflight_bx3, material_bx3x3=tfmat, shininess_bx1=tfshi)
       
        image = predictions.detach().cpu().numpy()[0]
        imageio.imwrite('results/example.png', (255*image[:,:,:]).astype(np.uint8))
        writer.append_data((image * 255).astype(np.uint8))

    writer.close()
    
   
    '''os.makedirs(data_dir_1,exist_ok=True)
    for num, azimuth in enumerate(theta2):
        # rest mesh to initial state
        #mesh.reset_()
        theta2.set_description('Drawing rotation')#*loop=theta2
        #renderer.transform.set_eyes_from_angles(camera_distance, theta1[num], azimuth)#elevation=theta1
        renderer.set_look_at_parameters([azimuth],
                                        [theta1[num]],
                                        [CAMERA_DISTANCE]) 
        predictions, _, _ = renderer(points=[vertices, faces[0].long()], colors=[uvs, face_textures[0], textures], 
                                     light=tflight_bx3, material=tfmat, shininess=tfshi)
        image = predictions.detach().cpu().numpy()[0]
        #image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
        end_time = time.time()
#        imageio.imwrite(('%s/%s.jpg'%(data_dir,num)),(255*image[:,:,:3]).astype(np.uint8))
        imageio.imwrite(('%s/t1_%s_t2_%s.png'%(data_dir_1,theta1[num],azimuth)),(255*image[:,:,:]).astype(np.uint8))
#        views=open(data_dir_1 +'/'+'view.txt','a')
#        thetas=[theta1[num],azimuth]
#        for theta in thetas:
#            views.write('%s,'%theta)
#            views.write('\t')
#        views.write('\n')
#    views.close()'''

if __name__ == '__main__':
    main()
