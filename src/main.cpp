#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/slice.h>
#include <igl/boundary_loop.h>
#include <igl/octree.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/knn.h>
#include <igl/cat.h>
#include <unordered_set>


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

// landmark indices
VectorXi landmarks;
VectorXi landmarksT;

// intermediate constraints assembled in nra_prep()
VectorXi inter_ind; 
MatrixXd inter_pos;

//knn
MatrixXi KNN;

// keep track of already added constraint to avoid duplicates
// unordered set has a useful find fct
unordered_set<int> added_constraintsT;
unordered_set<int> added_constraints;

VectorXi boundaryT_ind;
MatrixXd boundaryT_pos;

VectorXi boundary_ind;
MatrixXd boundary_pos;

// octree stuff
vector<vector<int>> point_indices;
MatrixXi CH;
MatrixXd CN;
MatrixXd W;

// Implicitly smoothed array, #Vx3
Eigen::MatrixXd Vt_impLap;


bool prepared = false;
bool init_knn = false;
int it = 1;
int cnt = 0;
double threshold = 0.4;

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
    landmarks = get_landmarks("./data/person0__23landmarks");
    igl::slice(V, landmarks, 1, landmarks_pos);

    //landmarks on template
    landmarksT = get_landmarks("./data/headtemplate_23landmarks");
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


// Copied and modified for bigger number of constraints from assignment4
void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should contain the positions in the correct order.
  double lambda = 1.;
	std::vector<Eigen::Triplet<double> > tripletList;
	C.resize(indices.rows()*3, Vt.rows()*3);
	d.resize(3 * indices.size());
  cout << "size of indices vector: " << indices.size() << endl;
	for (int i = 0; i < indices.size(); i++) {
		tripletList.push_back(Eigen::Triplet<double>(i, indices(i), lambda*1.));
		tripletList.push_back(Eigen::Triplet<double>(indices.rows()+i, Vt.rows()+indices(i), lambda*1.));
    tripletList.push_back(Eigen::Triplet<double>(2 * indices.rows()+i, 2 * Vt.rows()+indices(i), lambda*1.));
		d(i) = lambda*positions(i, 0);
		d(i + indices.rows()) = lambda*positions(i, 1);
    d(i + 2 * indices.rows()) = lambda*positions(i, 2);
	}
	C.setFromTriplets(tripletList.begin(), tripletList.end());
}


void nra_prep() {
  igl::boundary_loop(F, boundary_ind);
  igl::slice(V, boundary_ind, 1, boundary_pos);

  // add scan landmarks and boundary as constraints
  for (int i=0; i<boundary_ind.rows(); i++) {
      added_constraints.insert(boundary_ind(i));
  }

  for (int i=0; i<landmarks.rows(); i++) {
    added_constraints.insert(landmarks(i));
  }
  cout << "reached scan boundary" << endl;

  igl::boundary_loop(Ft, boundaryT_ind);
  igl::slice(Vt, boundaryT_ind, 1, boundaryT_pos);


  // add template landmarks and boundary as constraint
  for (int i=0; i<landmarksT.rows(); i++) {
    added_constraintsT.insert(landmarksT(i));
  }

  for (int i=0; i<boundary_ind.rows(); i++) {
    added_constraintsT.insert(boundaryT_ind(i));
  }

  cout << "reached template boundary" << endl;
  //add neighbors of boundary points
  // vector<vector<int>> Adj;
  // igl::adjacency_list(F, Adj);
  // int nb_neighbors = 1;
  // while(nb_neighbors > 0) {
  //     for (int i : added_constraints) {
  //         for (int neighbor : Adj[i]) {
  //             added_constraints.insert(neighbor);
  //             //cout << "added neigbor: " << neighbor << endl;
  //         }
  //     }
  //     nb_neighbors--;
  // }   

  // igl::cat(1, landmarksT, boundaryT_ind, inter_ind);
  // igl::cat(1, landmarks_pos, boundaryT_pos, inter_pos);
  inter_ind = landmarksT;
  inter_pos = landmarks_pos;

  cout << "reached neighbors to boundary" << endl;

  // setup octree
  igl::octree(V, point_indices, CH, CN, W);

  cout << "reached octree" << endl;
}


void non_rigid_alignment(int it=1) {
  if (it == 10) {
    // igl::cat(1, landmarksT, boundaryT_ind, inter_ind);
    // igl::cat(1, landmarks_pos, boundaryT_pos, inter_pos);
  }
  
  VectorXi nearest_ind(Vt.rows());
  MatrixXd nearest_pos(Vt.rows(), 3);
  if (it > 1) {
    // calculate knn
    int k = 20;
    if (!init_knn) {
      igl::knn(Vt, V, k, point_indices, CH, CN, W, KNN);
      cout << "reached knn" << endl;
      init_knn = true;
    }
    
    // add nearest neighbors to constraints
    int cnt = 0;
    int ind;
    for (int i = 0; i < Vt.rows(); i++) {
        ind = KNN(i, 1);
        double dist = (Vt.row(i) - V.row(ind)).norm();
        // check if point already in added_constraints
        if (added_constraintsT.find(i) == added_constraintsT.end() && added_constraints.find(ind) == added_constraints.end() && dist < threshold) {
          //cout << "added vertex: " << ind << endl;
          nearest_ind(cnt) = i;
          nearest_pos.row(cnt) = V.row(ind);
          added_constraints.insert(ind);
          cnt++;
        }
    }

    nearest_ind.conservativeResize(cnt);
    nearest_pos.conservativeResize(cnt, 3);
    cout << "reached nearest pts" << endl;
  }

  //----------------------------------------------------------------
  // ***************************************************************
  // ---------------------------------------------------------------
  // More or less same as in assignment4 but with more constraints
  SparseMatrix<double> A, C, L;
  VectorXd b(Vt.rows() * 3), d, x_prime;

  // L_cot*x_prime = L_cot*x
  igl::cotmatrix(Vt, Ft, L);
  b << L * Vt.col(0), L * Vt.col(1), L * Vt.col(2);
  igl::repdiag(L, 3, A);
  cout << "reached your favorite laplacian" << endl;

  VectorXi constraints_ind;
  MatrixXd constraints_pos;
  
  if (init_knn) {
    igl::cat(1, nearest_ind, inter_ind, constraints_ind);
    igl::cat(1, nearest_pos, inter_pos, constraints_pos);
  }
  else {
    constraints_ind = inter_ind;
    constraints_pos = inter_pos;
  }
  cout << "reached constraint_pts" << endl;
  ConvertConstraintsToMatrixForm(constraints_ind, constraints_pos, C, d);
  cout << "reached converted constraints to matrix" << endl;

  SparseMatrix<double> CT = C.transpose();
  SparseMatrix<double> zeros(C.rows(), CT.cols()/*C.rows()*/);
  SparseMatrix<double> LHS, inter1, inter2;
  VectorXd RHS; 

  zeros.setZero();
  igl::cat(2, A, CT, inter1);
  igl::cat(2, C, zeros, inter2);
  igl::cat(1, inter1, inter2, LHS);
  igl::cat(1, b, d, RHS);

  Eigen::SparseLU <Eigen::SparseMatrix<double>> solver;
  LHS.makeCompressed();
  solver.compute(LHS);
  cout << "reached prefactor system" << endl;
  x_prime = solver.solve(RHS);
  cout << "size 3*Vt.rows(): " << 3*Vt.rows() << endl;
  cout << "size x_prime: " << x_prime.size() << endl; 

  Vt.col(0) = x_prime.segment(0, Vt.rows());
  Vt.col(1) = x_prime.segment(Vt.rows(), Vt.rows());
  Vt.col(2) = x_prime.segment(2*Vt.rows(), Vt.rows());

  // update landmarkT positions and inter_pos
  // VectorXi boundaryT_ind;
  // MatrixXd boundaryT_pos;

  // igl::boundary_loop(Ft, boundaryT_ind);
  // igl::slice(Vt, boundaryT_ind, 1, boundaryT_pos);
  // igl::cat(1, landmarks_pos, boundaryT_pos, inter_pos);
  cout << "reached end of non-rigid-alignment" << endl;

}

void smooth() {
  double step = 0.000008;
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(Vt, Ft, L);
  cout << "smoothing cot" << endl;
  // Recompute just mass matrix on each step
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(Vt_impLap, Ft, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  cout << "smoothing mass" << endl;
  // Solve (M-delta*L) U = M*U
  const auto& S = (M - step * L);
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
  assert(solver.info() == Eigen::Success);
  Vt_impLap = solver.solve(M * Vt_impLap).eval();
  cout << "smoothing solve" << endl;
  //cout << Vt_impLap << endl;
  //Compute centroid and subtract (also important for numerics)
  Eigen::VectorXd dblA;
  igl::doublearea(Vt_impLap, Ft, dblA);
  cout << "smoothing area" << endl;
  double area = 0.5 * dblA.sum();
  Eigen::MatrixXd BC;
  igl::barycenter(Vt_impLap, Ft, BC);
  cout << "smoothing BC" << endl;
  Eigen::RowVector3d centroid(0, 0, 0);
  for (int i = 0; i < BC.rows(); i++)
  {
      centroid += 0.5 * dblA(i) / area * BC.row(i);
  }
  cout << "reached centroid" << endl;
  Vt_impLap.rowwise() -= centroid;
  //// Normalize to unit surface area (important for numerics)
  Vt_impLap.array() /= sqrt(area);
  cout << "end of smooth()" << endl;
}



bool load_mesh(string filename)
{
  igl::read_triangle_mesh(filename,V,F);
  viewer.data().clear();
  viewer.data().set_mesh(V, F);

  //viewer.core.align_camera_center(V); // incompatible with newer igl version
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
  //viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL); // incompatible with newer igl version
  viewer.launch();
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{

  if (key == '1') {
    rigid_alignment();
  }

  if (key == '2') {
    if (!prepared) {
      nra_prep();
      cout << "preparation for non_rigid_alignment done!" << endl;
      prepared = true;
    }
    non_rigid_alignment(it);
    it++;
    threshold += 0.3;
    Vt_impLap = Vt;
  }


  // Display alignment result
  MatrixXd VVt(V.rows() + Vt.rows(), 3);
  MatrixXi FFt(F.rows() + Ft.rows(), 3);
  VVt << V, Vt;
  FFt << F, Ft + MatrixXi::Constant(Ft.rows(), 3, V.rows()); // Need to add #V to change vertex indices of template faces
  viewer.data().clear();
  viewer.data().set_mesh(Vt, Ft);



  if (key == '3') {
    smooth();
    cout << "reached end of smoothing" << endl;
    viewer.data().clear();
    viewer.data().set_mesh(Vt_impLap, Ft);
    //viewer.core.align_camera_center(V_impLap, F);
  }
  return true;
}