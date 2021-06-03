// I couldn't get file_dialog_open to work on macOS, so I'm using the following workaround.
// For the code to work, there needs to exist a file meshlist.txt in the project folder.
// This file has to contain the names of all meshes and landmark files that the user wants
// to be able to load.
// The name of each of these files should be in a separate line (e.g. alain_normal.obj).
// On mac at least, this file can be easily created by running
// ls > ../meshlist.txt
// from the folder containing the meshes and landmark files.
// It is assumed that this folder is named "data" (without quotes). Otherwise, the variable
// pathtomeshes (defined below) has to be changed.

// Once this file is created, it is possible to go through all the meshes in the list
// linearly by pressing "Load next mesh" or "Load previous mesh" buttons.
// The original meshlist.txt contains the names of all meshes and landmark files from
// all_data/scanned_faces_cleaned

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>


#include "Lasso.h"
#include "Colors.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/slice.h>
#include <igl/adjacency_list.h>
#include <igl/grad.h>
#include <igl/doublearea.h>
#include <igl/diag.h>
#include <igl/repdiag.h>

#include <igl/writeOBJ.h>

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>

#include "Lasso.h"
#include "Colors.h"

//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;
using Viewer = igl::opengl::glfw::Viewer;


Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3), V_cp(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0, 3);

//mouse interaction
enum MouseMode { AREA, SINGLE };
MouseMode mouse_mode = AREA;
bool doit = false;
bool meshLoaded = false;
int down_mouse_x = -1, down_mouse_y = -1;

//for selecting vertices
std::unique_ptr<Lasso> lasso;
//list of currently selected vertices
Eigen::VectorXi selected_v(0, 1);

//for saving constrained vertices
//vertex-to-handle index, #V x1 (-1 if vertex is free)
Eigen::VectorXi handle_id(0, 1);
//list of all vertices belonging to handles, #HV x1
Eigen::VectorXi handle_vertices(0, 1);
//centroids of handle regions, #H x1
Eigen::MatrixXd handle_centroids(0, 3);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0, 3);
//index of handle being moved
int moving_handle = -1;
//rotation and translation for the handle being moved
Eigen::Vector3f translation(0, 0, 0);
Eigen::Vector4f rotation(0, 0, 0, 1.);
typedef Eigen::Triplet<double> T;
//per vertex color array, #V x3
Eigen::MatrixXd vertex_colors;

std::vector<pair<int, int>> landmarks;

Eigen::MatrixXd landmark_positions(0, 3), b;
Eigen::VectorXi not_handle_vertices(0, 1);
Eigen::SparseMatrix<double> L, M, MInverse, LMinverseL, AFF, AFC;
std::string latestMeshFileLoaded, latestLandmarkFileLoaded;

//function declarations (see below for implementation)
bool solve(Viewer& viewer);
void get_new_handle_locations();
Eigen::Vector3f computeTranslation(Viewer& viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
Eigen::Vector4f computeRotation(Viewer& viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
void compute_handle_centroids();
Eigen::MatrixXd readMatrix(const char* filename);

bool callback_mouse_down(Viewer& viewer, int button, int modifier);
bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(Viewer& viewer, int button, int modifier);
bool callback_pre_draw(Viewer& viewer);
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);
void onNewHandleID();
void applySelection();

void smooth()
{
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
    igl::invert_diag(M, MInverse);

    LMinverseL = L * MInverse * L;

    igl::slice(LMinverseL, not_handle_vertices, not_handle_vertices, AFF);
    igl::slice(LMinverseL, not_handle_vertices, handle_vertices, AFC);

    // As in the slides, right hand side will be -AFC * handle_vertices
    b = -AFC * handle_vertex_positions;

    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>, Eigen::RowMajor> solver;
    solver.compute(AFF); // Left hand side of our system to solve

    // And we solve for our smoothed new mesh
    Eigen::MatrixXd V_Smooth = solver.solve(b);

    // Now we have B, the smoothed non-handles and the correctly located handles
    igl::slice_into(V_Smooth, not_handle_vertices, 1, V);
    igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);

    viewer.data().set_mesh(V, F);
};

void get_new_handle_locations()
{

}

bool load_mesh(string filename)
{
    igl::read_triangle_mesh(filename, V, F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    viewer.core.align_camera_center(V);
    V_cp = V;
    handle_id.setConstant(V.rows(), 1, -1);
    // Initialize selector
    lasso = std::unique_ptr<Lasso>(new Lasso(V, F, viewer));

    selected_v.resize(0, 1);

    return true;
}


std::vector<string> meshlist;
std::vector<string> landmarklist;

std::vector<string>::iterator current_mesh;
bool first_init;
std::string pathtomeshes = "../data/"; // eg. ../data/


// updatemeshlist reads the file specified by meshlistpath and puts all the .obj files
// into meshlist vector and all the .landmark files into the landmarklist vector.
void updatemeshlist () {
    meshlist.clear();
    landmarklist.clear();

    for (const auto& entry : filesystem::directory_iterator(pathtomeshes)) //this requires c++ 17 but the alternatives are ugly as hell :)
    {
        std::string currentPath = entry.path().string();
        std::string file = currentPath.substr(currentPath.find_last_of("/") + 1);

        std::string extension = file.substr(file.find_last_of(".") + 1);
        std::string fileName = file.substr(0, file.find_last_of("."));

        if (extension == "obj") 
        {
            meshlist.push_back(fileName);
        }
        else if (extension == "landmark") 
        {
            landmarklist.push_back(fileName);
        }
        else 
        {
            cout << "File: " << currentPath << " not supported for landmarkins or smoothing." << endl;
        }
    }

    std::cout << "Total number of meshes: " << meshlist.size()
            << "\nNumber of landmark files: " << landmarklist.size()
              << std::endl;

    if (!first_init) {
        current_mesh = find(meshlist.begin(), meshlist.end(), *current_mesh);
    }

}

int main(int argc, char* argv[])
{
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    // Prevents weird crashes
    load_mesh("");

    first_init = 1;
    updatemeshlist();
    first_init = 0;

    current_mesh = meshlist.begin();

    latestMeshFileLoaded = *current_mesh;
    latestMeshFileLoaded.append(".obj");
    latestMeshFileLoaded.insert(0,pathtomeshes);
    load_mesh(latestMeshFileLoaded);
    std::cout << "Loaded mesh " << latestMeshFileLoaded << std::endl;


    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        int mouse_mode_type = static_cast<int>(mouse_mode);


        if (ImGui::CollapsingHeader("Landmarks", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Smoothing and selection");
            if (ImGui::Combo("Selection", &mouse_mode_type, "AREA\0SINGLE\0"))
            {
                mouse_mode = static_cast<MouseMode>(mouse_mode_type);
            }
            if (ImGui::Button("Apply Selection", ImVec2(-1, 0)))
            {
                applySelection();
            }
            if (ImGui::Button("Clear Constraints", ImVec2(-1, 0)))
            {
                handle_id.setConstant(V.rows(), 1, -1);
            }
            if (ImGui::Button("Smooth", ImVec2(-1, 0)))
            {
                smooth();
            }

            ImGui::Text("Landmarks");
            if (ImGui::Button("Remove last landmark", ImVec2(-1, 0)))
            {
                if (landmarks.size() > 0) {
                    // Remove last landmark
                    pair<int, int> landmark = landmarks.back();
                    handle_id[landmark.first] = -1;
                    landmarks.pop_back();
                    landmark_positions.conservativeResize(landmarks.size(), 3);

                    // Update points
                    viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                    // Remove latest label
                    viewer.data().labels_positions.conservativeResize(landmarks.size(), 3);
                    viewer.data().labels_strings.pop_back();
                }
            }

            if (ImGui::Button("Remove all landmarks", ImVec2(-1, 0)))
            {
                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                // Update points
                viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();

                handle_id.setConstant(V.rows(), 1, -1);
            }

            ImGui::Text("Files and mesh");
            // ToDo - Find a way to select folder and go through iteratively
            // if (ImGui::Button("Load new mesh", ImVec2(-1, 0)))
            if (ImGui::Button("Load next mesh", ImVec2(-1, 0)))
            {
                // latestMeshFileLoaded = igl::file_dialog_open();

                if (current_mesh < meshlist.end()) {
                    current_mesh++;
                }
                else {
                    std::cout << "There are no more meshes. " << std::endl;
                }

                latestMeshFileLoaded = *current_mesh;
                latestMeshFileLoaded.append(".obj");
                latestMeshFileLoaded.insert(0,pathtomeshes);
                load_mesh(latestMeshFileLoaded);

                std::cout << "Loaded mesh " << latestMeshFileLoaded << std::endl;

                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                handle_id.setConstant(V.rows(), 1, -1);

                // Update points
                viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();
            }
            if (ImGui::Button("Load previous mesh", ImVec2(-1, 0)))
            {
                // latestMeshFileLoaded = igl::file_dialog_open();

                if (current_mesh > meshlist.begin()) {
                    current_mesh--;
                }
                else {
                    std::cout << "There are no more meshes. " << std::endl;
                }

                latestMeshFileLoaded = *current_mesh;
                latestMeshFileLoaded.append(".obj");
                latestMeshFileLoaded.insert(0,pathtomeshes);
                load_mesh(latestMeshFileLoaded);

                std::cout << "Loaded mesh " << latestMeshFileLoaded << std::endl;

                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                handle_id.setConstant(V.rows(), 1, -1);

                // Update points
                viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();
            }
            if (ImGui::Button("Load next unprocessed mesh", ImVec2(-1, 0)))
            {
                // latestMeshFileLoaded = igl::file_dialog_open();
                //
                if (current_mesh < meshlist.end()) {
                    current_mesh++;
                }
                else {
                    std::cout << "There are no more meshes. " << std::endl;
                }

                for (auto landmark = landmarklist.begin(); landmark < landmarklist.end(); ++landmark) {
                    if (*landmark != *current_mesh)
                        continue;
                    else {
                        if (current_mesh < meshlist.end()) {
                            current_mesh++;
                            landmark = landmarklist.begin();
                        }
                        else {
                            std::cout << "All meshes are processed." << std::endl;
                        }
                    }
                }
                latestMeshFileLoaded = *current_mesh;
                latestMeshFileLoaded.append(".obj");
                latestMeshFileLoaded.insert(0,pathtomeshes);
                load_mesh(latestMeshFileLoaded);


                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                handle_id.setConstant(V.rows(), 1, -1);

                // Update points
                viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();
            }

            if (ImGui::Button("Load landmark file", ImVec2(-1, 0)))
            {
                updatemeshlist();

                string current_mesh_name = *current_mesh;
                if (find(landmarklist.begin(), landmarklist.end(), *current_mesh)!=landmarklist.end()) {
                    latestLandmarkFileLoaded = *current_mesh; //igl::file_dialog_open();
                    latestLandmarkFileLoaded.append(".landmark");
                    latestLandmarkFileLoaded.insert(0,pathtomeshes);
                    std::cout << "Loaded the landmark file " << latestLandmarkFileLoaded << std::endl;
                }
                else if (find(landmarklist.begin(),
                              landmarklist.end(),
                              current_mesh_name.substr(0, current_mesh_name.length()-9))
                              !=landmarklist.end()) {
                    latestLandmarkFileLoaded = current_mesh_name.substr(0, current_mesh_name.length()-9); //igl::file_dialog_open();
                    latestLandmarkFileLoaded.append(".landmark");
                    latestLandmarkFileLoaded.insert(0,pathtomeshes);
                    std::cout << "Loaded the landmark file " << latestLandmarkFileLoaded << std::endl;
                }
                else {
                    std::cout << "Can't find the landmark file. " << std::endl;
                }

                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();

                handle_id.setConstant(V.rows(), 1, -1);


                // Load in now landmarks
                string landmark;
                ifstream landmarkFile(latestLandmarkFileLoaded);
                if (landmarkFile.is_open())
                {
                    while (getline(landmarkFile, landmark))
                    {
                        int spaceIndex = landmark.find(" ");
                        string vertexIndex = landmark.substr(0, spaceIndex);
                        string landmarkIndex = landmark.substr(spaceIndex + 1, landmark.length());
                        int vertexIndexInt = stoi(vertexIndex);
                        int landmarkIndexInt = stoi(landmarkIndex);
                        landmarks.push_back(std::make_pair(vertexIndexInt, landmarkIndexInt));

                        handle_id[vertexIndexInt] = landmarkIndexInt;

                        landmark_positions.conservativeResize(landmarks.size(), 3);
                        landmark_positions.row(landmarks.size() - 1) = V.row(vertexIndexInt);

                        viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));
                        viewer.data().add_label(V.row(vertexIndexInt), to_string(landmarks.size()));
                    }
                    landmarkFile.close();
                }
            }

            if (ImGui::Button("Save landmarks", ImVec2(-1, 0)))
            {
                // Credits to https://stackoverflow.com/questions/4643512/replace-substring-with-another-substring-c - based on their implementation

                int endingIndex = -1;
                endingIndex = latestMeshFileLoaded.find(".obj");

                std::string fileToSaveLandmarks = latestMeshFileLoaded;
                fileToSaveLandmarks.replace(endingIndex, 4, ".landmark");

                ofstream landmarkFileStream(fileToSaveLandmarks);
                for (const pair<int, int>& landmark : landmarks)
                {
                    landmarkFileStream << landmark.first << " " << landmark.second << "\n";
                }

                landmarkFileStream.close();

                // cout << "Files succesfully saved to: " + fileToSaveSmoothed << " and " << fileToSaveLandmarks << endl;
                cout << "File succesfully saved to: " << fileToSaveLandmarks << endl;
            }
            if (ImGui::Button("Save mesh and landmarks", ImVec2(-1, 0)))
            {
                // Credits to https://stackoverflow.com/questions/4643512/replace-substring-with-another-substring-c - based on their implementation

                int endingIndex = -1;
                endingIndex = latestMeshFileLoaded.find(".obj");

                if (endingIndex < 0)
                    return;

                std::string fileToSaveSmoothed = latestMeshFileLoaded;
                fileToSaveSmoothed.replace(endingIndex, 4, "-smoothed.obj");

                igl::writeOBJ(fileToSaveSmoothed, V, F);


                std::string fileToSaveLandmarks = latestMeshFileLoaded;
                fileToSaveLandmarks.replace(endingIndex, 4, ".landmark");

                ofstream landmarkFileStream(fileToSaveLandmarks);
                for (const pair<int, int>& landmark : landmarks)
                {
                    landmarkFileStream << landmark.first << " " << landmark.second << "\n";
                }

                landmarkFileStream.close();

                cout << "Files succesfully saved to: " + fileToSaveSmoothed << " and " << fileToSaveLandmarks << endl;
            }
        }
    };


    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up = callback_mouse_up;
    viewer.callback_pre_draw = callback_pre_draw;

    viewer.data().point_size = 10;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();
}


bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int)Viewer::MouseButton::Right)
        return false;

    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    int vi = lasso->pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);

    if (vi < 0)
        return false;

    if (mouse_mode == AREA)
    {
        if (lasso->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y) >= 0)
            doit = true;
        else
            lasso->strokeReset();
    }
    else
    {
        handle_id[vi] = vi;

        int numFree = (handle_id.array() == -1).cast<int>().sum();
        int num_handle_vertices = V.rows() - numFree;
        handle_vertices.setZero(num_handle_vertices);
        handle_vertex_positions.setZero(num_handle_vertices, 3);
        // Quickly store which vertices are not handles for later on (simplify it)
        not_handle_vertices.setZero(V.rows() - num_handle_vertices);

        int count1 = 0;
        int count2 = 0;
        for (long vx = 0; vx < V.rows(); ++vx) {
            if (handle_id[vx] >= 0) {
                handle_vertices[count1] = vx;
                handle_vertex_positions.row(count1++) = V.row(vx);
            }
            else {
                not_handle_vertices[count2++] = vx;
            }
        }

        // Make sure we have not saved the same vertex as a landmark before
        for (const pair<int, int>& landmark : landmarks)
        {
            if (landmark.first == vi)
                return false;
        }
        landmarks.push_back(std::make_pair(vi, landmarks.size() + 1));

        landmark_positions.conservativeResize(landmarks.size(), 3);
        landmark_positions.row(landmarks.size() - 1) = V.row(vi);

        viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));
        viewer.data().add_label(V.row(vi), to_string(landmarks.size()));

    }
    return doit;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
    if (!doit)
        return false;
    if (mouse_mode == AREA)
    {
        lasso->strokeAdd(mouse_x, mouse_y);
        return true;
    }
    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier)
{
    if (!doit)
        return false;
    doit = false;
    if (mouse_mode == AREA)
    {
        selected_v.resize(0, 1);
        lasso->strokeFinish(selected_v);
        return true;
    }

    return false;
};


bool callback_pre_draw(Viewer& viewer)
{
    // initialize vertex colors
    vertex_colors = Eigen::MatrixXd::Constant(V.rows(), 3, .9);

    // first, color constraints
    int num = handle_id.maxCoeff();
    if (num == 0)
        num = 1;
    for (int i = 0; i < V.rows(); ++i)
        if (handle_id[i] != -1)
        {
            int r = handle_id[i] % MAXNUMREGIONS;
            vertex_colors.row(i) << regionColors[r][0], regionColors[r][1], regionColors[r][2];
        }
    // then, color selection
    for (int i = 0; i < selected_v.size(); ++i)
        vertex_colors.row(selected_v[i]) << 131. / 255, 131. / 255, 131. / 255.;

    viewer.data().set_colors(vertex_colors);
    viewer.data().V_material_specular.fill(0);
    viewer.data().V_material_specular.col(3).fill(1);
    viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE | igl::opengl::MeshGL::DIRTY_SPECULAR;


    //clear points and lines
    viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));
    viewer.data().set_edges(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXi::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));

    //draw the stroke of the selection
    for (unsigned int i = 0; i < lasso->strokePoints.size(); ++i)
    {
        viewer.data().add_points(lasso->strokePoints[i], Eigen::RowVector3d(0.4, 0.4, 0.4));
        if (i > 1)
            viewer.data().add_edges(lasso->strokePoints[i - 1], lasso->strokePoints[i], Eigen::RowVector3d(0.7, 0.7, 0.7));
    }

    // update the vertex position all the time
    viewer.data().V.resize(V.rows(), 3);
    viewer.data().V << V;

    viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_POSITION;

    return false;

}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{
    bool handled = false;
    if (key == 'A')
    {
        applySelection();
        callback_key_down(viewer, '1', 0);
        handled = true;
    }

    return handled;
}

void onNewHandleID()
{
    //store handle vertices too
    int numFree = (handle_id.array() == -1).cast<int>().sum();
    int num_handle_vertices = V.rows() - numFree;
    handle_vertices.setZero(num_handle_vertices);
    handle_vertex_positions.setZero(num_handle_vertices, 3);
    not_handle_vertices.setZero(V.rows() - num_handle_vertices);


    int count1 = 0;
    int count2 = 0;
    for (long vi = 0; vi < V.rows(); ++vi) {
        if (handle_id[vi] >= 0) {
            handle_vertices[count1] = vi;
            handle_vertex_positions.row(count1++) = V.row(vi);
        }
        else {
            not_handle_vertices[count2++] = vi;
        }
    }

    compute_handle_centroids();
}

void applySelection()
{
    int index = handle_id.maxCoeff() + 1;
    for (int i = 0; i < selected_v.rows(); ++i)
    {
        const int selected_vertex = selected_v[i];
        if (handle_id[selected_vertex] == -1)
            handle_id[selected_vertex] = index;
    }
    selected_v.resize(0, 1);

    onNewHandleID();
}

void compute_handle_centroids()
{
    //compute centroids of handles
    int num_handles = handle_id.maxCoeff() + 1;
    handle_centroids.setZero(num_handles, 3);

    Eigen::VectorXi num; num.setZero(num_handles, 1);
    for (long vi = 0; vi < V.rows(); ++vi)
    {
        int r = handle_id[vi];
        if (r != -1)
        {
            handle_centroids.row(r) += V.row(vi);
            num[r]++;
        }
    }

    for (long i = 0; i < num_handles; ++i)
        handle_centroids.row(i) = handle_centroids.row(i).array() / num[i];

}



























/*
//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;
using namespace Eigen;

using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0,3);

//mouse interaction
int down_mouse_x = -1, down_mouse_y = -1;

//for selecting vertices
std::unique_ptr<Lasso> lasso;







bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int)Viewer::MouseButton::Right)
        return false;
    if (V.rows() < 1)
        return false;

    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    if (mouse_mode == AREA)
    {
        if (lasso->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y) >= 0)
            doIt = true;
        else
            lasso->strokeReset();
    }
    else
    {
        int vi = lasso->pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);

        if (vi < 0)
            return false;

        handle_id[vi] = vi;

        int numFree = (handle_id.array() == -1).cast<int>().sum();
        int num_handle_vertices = V.rows() - numFree;
        handle_vertices.setZero(num_handle_vertices);
        handle_vertex_positions.setZero(num_handle_vertices, 3);
        // Quickly store which vertices are not handles for later on (simplify it)
        not_handle_vertices.setZero(V.rows() - num_handle_vertices);

        int count1 = 0;
        int count2 = 0;
        for (long vx = 0; vx < V.rows(); ++vx) {
            if (handle_id[vx] >= 0) {
                handle_vertices[count1] = vx;
                handle_vertex_positions.row(count1++) = V.row(vx);
            }
            else {
                not_handle_vertices[count2++] = vx;
            }
        }

        // Make sure we have not saved the same vertex as a landmark before
        for (const pair<int, int>& landmark : landmarks)
        {
            if (landmark.first == vi)
                return false;
        }
        landmarks.push_back(std::make_pair(vi, landmarks.size() + 1));

        landmark_positions.conservativeResize(landmarks.size(), 3);
        landmark_positions.row(landmarks.size() - 1) = V.row(vi);

        viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));
        viewer.data().add_label(V.row(vi), to_string(landmarks.size()));

    }

    return true;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
    if (!doIt)
        return false;
    if (mouse_mode == AREA)
    {
        lasso->strokeAdd(mouse_x, mouse_y);
        return true;
    }
    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier)
{
    if (!doIt)
        return false;
    doIt = false;
    if (mouse_mode == AREA)
    {
        selected_v.resize(0, 1);
        lasso->strokeFinish(selected_v);
        return true;
    }
    return false;
};

bool callback_pre_draw(Viewer& viewer)
{
    // initialize vertex colors
    vertex_colors = Eigen::MatrixXd::Constant(V.rows(), 3, .9);

    // first, color constraints
    int num = handle_id.maxCoeff();
    if (num == 0)
        num = 1;
    for (int i = 0; i < V.rows(); ++i)
        if (handle_id[i] != -1)
        {
            int r = handle_id[i] % MAXNUMREGIONS;
            vertex_colors.row(i) << regionColors[r][0], regionColors[r][1], regionColors[r][2];
        }
    // then, color selection
    for (int i = 0; i < selected_v.size(); ++i)
        vertex_colors.row(selected_v[i]) << 131. / 255, 131. / 255, 131. / 255.;

    viewer.data().set_colors(vertex_colors);
    viewer.data().V_material_specular.fill(0);
    viewer.data().V_material_specular.col(3).fill(1);
    viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE | igl::opengl::MeshGL::DIRTY_SPECULAR;


    //clear points and lines
    viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));
    viewer.data().set_edges(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXi::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));

    //draw the stroke of the selection
    for (unsigned int i = 0; i < lasso->strokePoints.size(); ++i)
    {
        viewer.data().add_points(lasso->strokePoints[i], Eigen::RowVector3d(0.4, 0.4, 0.4));
        if (i > 1)
            viewer.data().add_edges(lasso->strokePoints[i - 1], lasso->strokePoints[i], Eigen::RowVector3d(0.7, 0.7, 0.7));
    }

    // update the vertex position all the time
    viewer.data().V.resize(V.rows(), 3);
    viewer.data().V << V;

    viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_POSITION;

    return false;
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{
  bool handled = false;

  // Could add shortcuts if people want them

  /*
  if (key == 'S')
  {
    mouse_mode = SELECT;
    handled = true;
  }

  if ((key == 'T') && (modifiers == IGL_MOD_ALT))
  {
    mouse_mode = TRANSLATE;
    handled = true;
  }

  if ((key == 'R') && (modifiers == IGL_MOD_ALT))
  {
    mouse_mode = ROTATE;
    handled = true;
  }
  if (key == 'A')
  {
    applySelection();
    callback_key_down(viewer, '1', 0);
    handled = true;
  }

  //viewer.ngui->refresh();
  return handled;
}
*/
