#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/slice.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0,3), V_cp(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0,3);

// headtemplate vertices and faces
MatrixXd Vt;
MatrixXi Ft;

// positions of landamrks for scan and template
MatrixXd landmarks_pos;
MatrixXd landmarksT_pos;

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

VectorXi get_landmarks(const char *filename)
{
    VectorXi landmarks;
    // no clue how many landmarks there are -- 100 should be enough
    landmarks.resize(100);

    fstream file(filename);
    int vid = 0; // current vertex
    int l = 0; // current landmark -- indexing starts at 1
    int size = 0; // #landmarks

    while(file >> vid >> l) {
        landmarks(l-1) = vid;
        size += 1;//max(size, lid);
    }

    landmarks.conservativeResize(size);
    return landmarks;
}

double mean_dist(MatrixXd pos, RowVector3d center) {
  return ((pos.rowwise() - center).colwise().norm()).mean();
}


void rigid_alignment(string obj_file="", string lm_file="" )
{
    //landmarks on scanned person
    VectorXi landmarks = get_landmarks("./data/person0__23landmarks");
    igl::slice(V, landmarks, 1, landmarks_pos);

    //landmarks on template
    VectorXi landmarksT = get_landmarks("./data/headtemplate_23landmarks");
    igl::slice(Vt, landmarksT, 1, landmarksT_pos);

    // center template to origin s.t. mean of its vertices is (0,0,0)
    RowVector3d mean_centerT = Vt.colwise().mean();
    Vt = Vt.rowwise() - mean_centerT;

    // Why would we rescale the scan? (Is being mentioned on ex. slide page 10)
    // rescale template s.t. average distance to mean landmark is the same for scan and template
    RowVector3d center_lm = landmarks_pos.colwise().mean();
    RowVector3d center_lmT = landmarksT_pos.colwise().mean();

    double mean_dist_lm = mean_dist(landmarks_pos, center_lm);
    double mean_dist_lmT = mean_dist(landmarksT_pos, center_lmT);
    // scale by ratio mean distance of scan landmarks and mean distance of template landmarks
    Vt = Vt*(mean_dist_lm / mean_dist_lmT);
    //update landmark template positions after rescale
    igl::slice(Vt, landmarksT, 1, landmarksT_pos);

    // center at mean of landmarks
    V = V.rowwise() - center_lm;
    Vt = Vt.rowwise() - center_lmT;
    // update landmark positions
    igl::slice(V, landmarks, 1, landmarks_pos);
    igl::slice(Vt, landmarksT, 1, landmarksT_pos);

    // rotation matrix via SVD
    Matrix3d Cov, Sign, U, Vsvd, R;

    Cov = landmarks_pos.transpose() * landmarksT_pos; //covariance
    JacobiSVD<Matrix3d> SVD(Cov, ComputeThinU | ComputeThinV);
    Sign = Matrix3d::Identity();
    U = SVD.matrixU();
    Vsvd = SVD.matrixV();
    //This broke my neck in assignment 4 --> if det negative flip sign of det (det of rot mat either 1 or -1)
    if((U * Vsvd.transpose()).determinant() < 0) {
      Sign << 1, 0, 0,
              0, 1, 0,
              0, 0, -1;
    }
    R = U * Sign * Vsvd.transpose();
    V = V*R;

    // update landmark position after the scan has been rotated
    igl::slice(V, landmarks, 1, landmarks_pos);
}

bool load_mesh(string filename)
{
  igl::read_triangle_mesh(filename,V,F);
  viewer.data().clear();
  viewer.data().set_mesh(V, F);

  viewer.core.align_camera_center(V);
  V_cp = V;
  return true;
}

int main(int argc, char *argv[])
{
  //load scanned person
  load_mesh("./data/person0_.obj");
  // load template
  igl::read_triangle_mesh("./data/headtemplate.obj", Vt, Ft);

  // For some reason I cannot dispaly the scan and template initially
  // MatrixXd VVt(V.rows() + Vt.rows(), 3);
  // MatrixXi FFt(F.rows() + Ft.rows(), 3);
  // VVt << V, Vt;
  // FFt << F, Ft + MatrixXi::Constant(Ft.rows(), 3, V.rows()); // Need to add #V to change vertex indices of template faces
  // viewer.data().clear();
  // viewer.data().set_mesh(VVt, FFt);
  // viewer.core.align_camera_center(VVt);

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]()
  {
    // Draw parent menu content
    menu.draw_viewer_menu();

    // Add new group
    if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen))
    {

    }
  };

  viewer.callback_key_down = callback_key_down;

  viewer.data().point_size = 10;
  viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.launch();
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{

  if (key == '1') {
    rigid_alignment();
    MatrixXd VVt(V.rows() + Vt.rows(), 3);
    MatrixXi FFt(F.rows() + Ft.rows(), 3);
    VVt << V, Vt;
    FFt << F, Ft + MatrixXi::Constant(Ft.rows(), 3, V.rows()); // Need to add #V to change vertex indices of template faces
    viewer.data().clear();
    viewer.data().set_mesh(VVt, FFt);
  }

  return true;
}