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


std::vector<pair<int, int>> landmarks;

Eigen::MatrixXd landmark_positions(0, 3);

std::string latestMeshFileLoaded, latestLandmarkFileLoaded;

bool callback_mouse_down(Viewer& viewer, int button, int modifier);
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

bool load_mesh(string filename)
{
  igl::read_triangle_mesh(filename,V,F);
  viewer.data().clear();
  viewer.data().set_mesh(V, F);

  viewer.core.align_camera_center(V);

  lasso = std::unique_ptr<Lasso>(new Lasso(V, F, viewer));

  landmarks.clear();

  return true;
}

int main(int argc, char *argv[])
{
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();
    
        if (ImGui::CollapsingHeader("Landmarks", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Files and mesh");
            // ToDo - Find a way to select folder and go through iteratively
            if (ImGui::Button("Load new mesh", ImVec2(-1, 0)))
            {
                latestMeshFileLoaded = igl::file_dialog_open();
                load_mesh(latestMeshFileLoaded);

                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                // Update points
                viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();
            }

            if (ImGui::Button("Load landmark file", ImVec2(-1, 0)))
            {
                latestLandmarkFileLoaded = igl::file_dialog_open();

                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();

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

                        landmark_positions.conservativeResize(landmarks.size(), 3);
                        landmark_positions.row(landmarks.size() - 1) = V.row(vertexIndexInt);

                        viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));
                        viewer.data().add_label(V.row(vertexIndexInt), to_string(landmarks.size()));
                    }
                    landmarkFile.close();
                }
            }

            if (ImGui::Button("Save mesh and landmarks", ImVec2(-1,0)))
            {
                // ToDo - save mesh líka

                // Credits to https://stackoverflow.com/questions/4643512/replace-substring-with-another-substring-c - based on their implementation

                int endingIndex = -1;
                endingIndex = latestMeshFileLoaded.find(".obj");

                if (endingIndex < 0)
                    return;

                std::string fileToSave = latestMeshFileLoaded;
                fileToSave.replace(endingIndex, 4, ".landmark");

                ofstream landmarkFileStream;
                landmarkFileStream.open(fileToSave);

                for (const pair<int, int>& landmark : landmarks)
                {
                    landmarkFileStream << landmark.first << " " << landmark.second << "\n";
                }

                landmarkFileStream.close();

                cout << "File succesfully saved to: " + fileToSave << endl;
            }
            ImGui::Text("Landmarks");
            if (ImGui::Button("Remove last landmark", ImVec2(-1,0)))
            {
                if (landmarks.size() > 0) {
                    // Remove last landmark
                    landmarks.pop_back();
                    landmark_positions.conservativeResize(landmarks.size(), 3);

                    // Update points
                    viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                    // Remove latest label
                    viewer.data().labels_positions.conservativeResize(landmarks.size(), 3);
                    viewer.data().labels_strings.pop_back();
                }
            }

            if (ImGui::Button("Remove all landmarks", ImVec2(-1,0)))
            {
                // Remove all landmarks
                landmarks.clear();
                landmark_positions.resize(0, 3);

                // Update points
                viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));

                // Remove all labels
                viewer.data().labels_positions.resize(0, 3);
                viewer.data().labels_strings.clear();
            }
        }
    };


  viewer.callback_key_down = callback_key_down;
  viewer.callback_mouse_down = callback_mouse_down;

  viewer.data().point_size = 10;
  viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.launch();
}


bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int)Viewer::MouseButton::Right)
        return false;
    if (V.rows() < 1)
        return false;

    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    int vi = lasso->pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);

    if (vi < 0)
        return false;

    // Make sure we have not saved the same vertex as a landmark before
    for (const pair<int, int> &landmark : landmarks)
    {
        if (landmark.first == vi)
            return false;
    }

    landmarks.push_back(std::make_pair(vi, landmarks.size() + 1));

    landmark_positions.conservativeResize(landmarks.size(), 3);
    landmark_positions.row(landmarks.size() - 1) = V.row(vi);

    viewer.data().set_points(landmark_positions, Eigen::RowVector3d(1, 0, 0));
    viewer.data().add_label(V.row(vi), to_string(landmarks.size()));

    return true;
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
  */

  //viewer.ngui->refresh();
  return handled;
}