#include <iostream>
#include <filesystem>
#include <string>

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

using namespace std;


using Viewer = igl::opengl::glfw::Viewer;
Viewer viewer;

// For display purposes only
Eigen::MatrixXd V_show;
Eigen::MatrixXi F_show;

std::vector<Eigen::MatrixXd> v_list;


const unsigned n_eigenvalues = 8;
double eigen_values_array[n_eigenvalues];

unsigned v_rows;
unsigned v_cols;
unsigned n_meshes;

Eigen::MatrixXd mean_face;
Eigen::MatrixXd A;
Eigen::MatrixXd PCA_U;
Eigen::MatrixXd PCA_s;
Eigen::MatrixXd PCA_V;

// Functions
void createGrid();
void load_mesh(string filename);
void load_all_meshes();
void compute_mean_face();
void compute_pca();
void change_eigen_value();
void display_eigen_face(unsigned i);



void display_eigen_face(unsigned i)
{
	V_show = PCA_U.col(i);

    double scaling_factor = V_show.rowwise().norm().mean();
    V_show /= scaling_factor;

    V_show += mean_face;

    V_show.resize(v_rows, v_cols);

	viewer.data().clear();
	viewer.data().set_mesh(V_show, F_show);
	viewer.core.align_camera_center(V_show, F_show);
}


void change_eigen_value()
{

	for (unsigned i = 0; i < n_eigenvalues; i++)
	{
		PCA_s(i, 0) = eigen_values_array[i];
	}

	Eigen::MatrixXd V_show1 = PCA_U * PCA_s;
	//V_show1 += mean_face;
	
	V_show.resize(v_rows, v_cols);
	for (unsigned row = 0; row < v_rows; ++row) {
		for (unsigned col = 0; col < v_cols; ++col) {
			V_show(row, col) = V_show1(col * v_rows + row);
		}
	}
	
	std::cout << V_show.rows() << " " << V_show.cols() << std::endl;
	viewer.data().clear();
	viewer.data().set_mesh(V_show, F_show);
	viewer.core.align_camera_center(V_show, F_show);
}

void compute_pca()
{
	compute_mean_face();
	A.resize(v_rows * v_cols, n_meshes);
	A.setZero();

	// setup the system
	for (unsigned i = 0; i < v_list.size(); i++)
	{
		//Eigen::MatrixXd v_temp = v_list[i] - mean_face;
		Eigen::MatrixXd v_temp = v_list[i];
		A.col(i) = v_temp.col(0);
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	PCA_s = svd.singularValues().head(n_eigenvalues); // this is a Vector
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();

	unsigned ns = PCA_s.rows();
	unsigned safeMax = min(n_eigenvalues, ns); // in case k > number sing. values

	PCA_U = U.leftCols(safeMax);
	//PCA_U = U.leftCols(safeMax).array().rowwise() *
	//	s.head(safeMax).transpose().array();
	PCA_V = V.leftCols(safeMax);

	for (unsigned i = 0; i < n_eigenvalues; i++)
	{
		eigen_values_array[i] = PCA_s(i, 0);
	}

	//std::cout << U.col(0) << std::endl;
	std::cout << "Mesh: " << v_rows << " " << v_cols << " " << n_meshes << std::endl;
	std::cout << "U: " << PCA_U.rows() << " " << PCA_U.cols() << std::endl;
	std::cout << "S: " << PCA_s.rows() << " " << PCA_s.cols() << std::endl;
	std::cout << "V: " << PCA_V.rows() << " " << PCA_V.cols() << std::endl;
	std::cout << "mean: " << mean_face.rows() << " " << mean_face.cols() << std::endl;


}

void compute_mean_face()
{
	mean_face.resize(v_rows * v_cols, 1);
	mean_face.setZero();

	for (unsigned i = 0; i < n_meshes; i++)
	{
		mean_face += v_list[i];
	}
	mean_face /= (double)n_meshes;
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {

	if (key == '1')
	{
		viewer.data().clear();
	}
	return true;
}


void load_all_meshes()
{
	n_meshes = 0;
	std::string path = "../data/";
	for (const auto& entry : filesystem::directory_iterator(path)) //this requires c++ 17 but the alternatives are ugly as hell :)
	{
		std::cout << entry.path() << std::endl;
		Eigen::MatrixXd v_temp;

		igl::read_triangle_mesh(entry.path().string(), v_temp, F_show);

		v_rows = v_temp.rows();
		v_cols = v_temp.cols();

		Eigen::VectorXd v_temp2(v_rows * v_cols);

		for (unsigned row = 0; row < v_rows; ++row) {
			for (unsigned col = 0; col < v_cols; ++col) {
				v_temp2(col * v_rows + row) = v_temp(row, col);
			}
		}

		v_list.push_back(v_temp2);

		n_meshes++;
	}
}


void load_mesh(string filename)
{
	igl::read_triangle_mesh(filename, V_show, F_show);
}



int main(int argc, char* argv[]) {

	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	viewer.callback_key_down = callback_key_down;

	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("Options", ImGuiTreeNodeFlags_DefaultOpen))
		{
			for (unsigned i = 0; i < n_eigenvalues; i++)
			{
				std::string i_str = "Eigenface " + std::to_string(i);
				char const* i_chr = i_str.c_str();
				if (ImGui::Button(i_chr, ImVec2(-1, 0)))
				{
					std::cout << "Load Eigenface " << i << std::endl;
					display_eigen_face(i);
				}
			}

			for (unsigned i = 0; i < n_eigenvalues; i++)
			{
				std::string i_str = "Eigenvalue " + std::to_string(i);
				char const* i_chr = i_str.c_str();
				if (ImGui::DragScalar(i_chr, ImGuiDataType_Double, &(eigen_values_array[i]), 0.1, 0, 0, "%.4f"))
				{
					std::cout << "Change Eigenvalue " << i << std::endl;
					change_eigen_value();
				}
			}
		}
	};
	load_all_meshes();
	compute_pca();

	callback_key_down(viewer, '1', 0);
	viewer.launch();
}
