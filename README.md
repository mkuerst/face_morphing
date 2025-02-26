# Shape Modeling and Geometry Processing - Assignment 6 Group Project - Group 6

## Student data

Name: Haraldur Orri Hauksson  
Legi-Nr: 20-942-892  
Email: hhauksson@student.ethz.ch

Name: Michael Kürsteiner  
Legi-Nr: 16-708-414  
Email: kumichae@student.ethz.ch

Name: Pavel Pozdnyakov  
Legi-Nr: 11-557-048  
Email: popavel@student.ethz.ch

Name: Ana Zegarac  
Legi-Nr: 18-941-591  
Email: ana.zegarac@math.ethz.ch

Name: Christopher Vogelsanger  
Legi-Nr: 16-933-863  
Email: cvogelsa@student.ethz.ch


## Work division
- **Landmark selection**: Haraldur Orri Hauksson, Ana Zegarac
- **Smoothing for mesh boundaries**: Haraldur Orri Hauksson
- **Rigid face alignment**: Michael Kürsteiner
- **Non-rigid face alignment**: Michael Kürsteiner, Haraldur Orri Hauksson, Pavel Pozdnyakov
- **PCA, UI**: Christopher Vogelsanger, Pavel Pozdnyakov, Ana Zegarac
- **Report, Presentation slides**: Ana Zegarac, Pavel Pozdnyakov
- **Bonus 2 (Autoencoder)**: Christopher Vogelsanger


## Presentation slides
Can be found in the `results` folder or on the following link:
https://docs.google.com/presentation/d/1TO7acJaVpoU-tNrBaVQhV4lMdOnlyNggBV0Yxc8fQ5o/edit?usp=sharing  

## Report

Since the project has three distinct parts, we organised the codebase in three branches:
- landmark;
- non-rigid-alignment; and
- pca.

We describe our approach to each of these three parts below and link the outputs.

### OS compatibility warning

Most parts of the project rely on the filesystem library to load the meshes iteratively. This library requires C++17. Moreover, while 
```
for (const auto& entry : filesystem::directory_iterator(pathtomeshes))
{
  ...
}
```
works on Windows, on macOS this line has to be changed to 
```
for (const auto& entry : std::__fs::filesystem::directory_iterator(pathtomeshes))
{
  ...
}
```

### Landmarks

The UI for creating landmarks looks as follows.
![](results/landmark-UI.png)

Core features are:
- Select landmarks and save them into .landmark files;
- If the user makes a mistake when creating landmarks, possibility to remove the last created landmark (or all of the created landmarks at once);
- Iteratively load (and process) meshes from the `data` folder;
- Smooth out the given mesh outside of the selected area.

Since alignment worked okay on the original meshes, we did not end up using the smoothing function.

#### Landmarks output

We created landmark files for 112 meshes from `scanned_faces_cleaned`. Each landmark file consists of 23 landmarks distributed as is shown in the screenshot below.

The landmark files can be found at https://polybox.ethz.ch/index.php/s/CSzukEdXSbtsdjh

#### Additional screenshots

##### Landmark creation
![](results/landmark-landmarks.png)

##### Smoothing out the meshes
![](results/landmark-selection.png)
![](results/landmark-selection1.png)
![](results/landmark-smoothing2.png)



### Alignment

The non-rigid-alignment code starts by loading a predetermined mesh. The user can then:
1. Perform rigid alignment by pressing `1`;
2. Perform non-rigid-alignment once by pressing `2` (each time an additional landmark is added into boundary conditions);
3. Perform smoothing by pressing `3`;
4. Save the aligned mesh;
5. Run steps 1, 2 and 4 for all the meshes in the folder `data`, with step 2 being executed until the mesh converges.

#### Alignment output

The aligned meshes can be found at https://polybox.ethz.ch/index.php/s/tDdk5dlYEw657HD

#### Video and screenshots
Non-rigid alignment is demonstrated in the video below:
https://youtu.be/sTtTwpT7WTU  

Alignment UI can be seen below.
![](results/alignment-UI.png)

Rigid alignment
![](results/alignment-rigid.png)

Non-rigid alignment (1st iteration)
![](results/alignment-nonrigid0.png)

Non-rigid alignment (after a couple iterations)
![](results/alignment-nonrigid1.png)

Non-rigid alignment (converged)
![](results/alignment-nonrigid2.png)

Non-rigid smoothed (converged)
![](results/alignmen-smoothed.png)

![Screenshot 2021-05-31 115452](https://user-images.githubusercontent.com/42907161/120681956-351d7f80-c49c-11eb-9e3b-b59c8184d5fc.png)  


#### Additional remarks: Small Isolated Components  

Two of the provided scanned and cleaned faces (alex wacky & nick smile) had a very small isolated component on top that would make the system we build for the non-rigid alignment unsolvable and the program would crash. Hence, we removed those isolated components in MeshLab and then ran the non-rigid alignment on the new meshes. The alex wacky mesh can now be non-rigidly aligned. However, the nick smile mesh crashes during the building process of its octree in the igl::octree call because of a stack overflow. Unfortunately, we ran out of time and were not able to find the cause of the problem.  

Download link to the scanned faces without these isolated components and the non-rigidly aligned alex wacky mesh:  
https://polybox.ethz.ch/index.php/s/d2ymit9koySCvi7

Below are the meshes with the isolated components loaded in MeshLab:  

<img width="589" alt="Screenshot 2021-06-03 at 20 07 21" src="https://user-images.githubusercontent.com/42907161/120756648-ce39ae00-c50f-11eb-90b5-45018a8b0119.png">  
<img width="534" alt="Screenshot 2021-06-03 at 20 07 43" src="https://user-images.githubusercontent.com/42907161/120756658-d1cd3500-c50f-11eb-844e-ff2c89c114a2.png">  

### PCA

The pca code runs PCA on all meshes in the folder `data` and allows the user to fine-tune 8 eigenfaces though the UI displayed below.
![](results/pca-UI.png)

To perform morphing from e.g. a face with a neutral facial expression to a face with a smile, the user can simply insert the two corresponding aligned .obj files into `data` and change eigenvalues accordingly. The next three screenshots showcase this progression from 'neutral' to 'smile'.
![](results/pca-smile0.png)
![](results/pca-smile1.png)
![](results/pca-smile2.png)

In the example below we perfomed PCA on on all faces with smile and neutral facial expression. 
One of the possible transfomations is illustrated by the two images below.
![](results/pca_morphing_non-smile.png)
![](results/pca_morphing_smile.png)

The next two images illustrate the possibility to morph one face into another.
![](results/pca_morphing_face_a.png) 
![](results/pca_morphing_face_b.png)

To be able to do that we performed the PCA on 6 different meshes, 3 for each face ("selina" and "jan").
The morphing occurs by increasing or decreasing the eigenvalues corresponding to the eigenfaces 
responsible for the representation of a particular face.

Below are the six eigenfaces. 
One can see that the first one represents the features of both of the faces.
![](results/pca_morphing_eigenface_0.png)

The second and third eigenfaces correspond more to the features of the first face.
![](results/pca_morphing_eigenface_1.png)
![](results/pca_morphing_eigenface_2.png)

The next three eigenfaces correspond more to the features of the second face.
![](results/pca_morphing_eigenface_3.png)
![](results/pca_morphing_eigenface_4.png)
![](results/pca_morphing_eigenface_5.png)

Otherwise, by running the code on all available meshes, the user can analyse the most prominent features in the given set of meshes and create new meshes. The UI also allows the user to save the newly created meshes.  

### Autoencoder

The autoencoder code runs a simple fully connected autoencoder on all meshes in the folder `data`. A seperate viewer allows the user to look at reconstructions and change the latent space. (Download link for trained networks, https://polybox.ethz.ch/index.php/s/8DPcxE043TmiCjq)

![](results/ae-0.PNG)

We tested around with different networks (Convolutional, VAEs) and multiple preprocessing steps (subtracting the mean, scaling the vertex positions down to 1.0, -1.0) but in the end a simple fully connected network gave the best results for the small set of last years data provided (see 2 sections down). For the higher resolution meshes of this year the network learns more of a mean mesh.

Reconstruction of a high resulution mesh:

![](results/ae-1.PNG)

Latent space impact:

![](results/ae-2.PNG)

Latent space impact:

![](results/ae-3.PNG)

Latent space impact:

![](results/ae-4.PNG)


The high number of verticies makes the compression rather steep and thus the reconstructed faces look all very similar. On the lower resolution data from the past years the approach works definitively better.

Reconstruction of a low resulution mesh:

![](results/ae-5.PNG)

Latent space impact:

![](results/ae-6.PNG)

Latent space impact:

![](results/ae-7.PNG)


