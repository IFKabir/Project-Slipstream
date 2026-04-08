#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "include/nlohmann/json.hpp"

// Cross-platform path separator and filesystem utilities
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <libgen.h>
#include <cstring>
#include <climits>
#endif

using json = nlohmann::json;

// Get the directory that the executable lives in
std::string get_exe_dir()
{
#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(NULL, buf, MAX_PATH);
    std::string path(buf);
    size_t pos = path.find_last_of("\\/");
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#else
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len == -1)
    {
        // Fallback: try current working directory
        return ".";
    }
    buf[len] = '\0';
    std::string path(buf);
    size_t pos = path.find_last_of('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#endif
}

struct TreeNode
{
    int feature_index;
    double threshold;
    int left_child;
    int right_child;
    bool is_leaf;
    double expected_finish;
};

class DecisionTree
{
public:
    std::vector<TreeNode> nodes;

    double predict(const std::vector<double> &features) const
    {
        int current_idx = 0;

        while (!nodes[current_idx].is_leaf)
        {
            const TreeNode &current = nodes[current_idx];
            if (features[current.feature_index] <= current.threshold)
            {
                current_idx = current.left_child;
            }
            else
            {
                current_idx = current.right_child;
            }
        }
        return nodes[current_idx].expected_finish;
    }
};

class RandomForest
{
public:
    std::vector<DecisionTree> trees;

    bool load_model(const std::string &filepath)
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open model file: " << filepath << "\n";
            return false;
        }

        json j;
        file >> j;

        for (const auto &tree_json : j["trees"])
        {
            DecisionTree tree;
            for (const auto &node_json : tree_json["nodes"])
            {
                TreeNode node;
                node.feature_index = node_json["feature"];
                node.threshold = node_json["threshold"];
                node.left_child = node_json["left"];
                node.right_child = node_json["right"];
                node.is_leaf = node_json["is_leaf"];
                node.expected_finish = node_json["prob"];

                tree.nodes.push_back(node);
            }
            trees.push_back(tree);
        }
        return true;
    }

    double predict(const std::vector<double> &features) const
    {
        if (trees.empty())
            return 0.0;

        double total_score = 0.0;
        for (const auto &tree : trees)
        {
            total_score += tree.predict(features);
        }
        return total_score / trees.size();
    }
};

struct Driver
{
    std::string name;
    std::vector<double> features;
    double predicted_finish;
};

bool compareDrivers(const Driver &a, const Driver &b)
{
    return a.predicted_finish < b.predicted_finish;
}

// Cross-platform path join
std::string path_join(const std::string &a, const std::string &b)
{
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    if (a.empty())
        return b;
    if (a.back() == sep || a.back() == '/')
        return a + b;
    return a + sep + b;
}

int main()
{
    // Resolve paths relative to the executable location (models/ directory)
    std::string exe_dir = get_exe_dir();
    // The exe lives in <project>/models/, so project root is one level up
    std::string project_root = path_join(exe_dir, "..");

    std::string model_path = path_join(exe_dir, "model_metadata.json");
    std::string grid_path = path_join(project_root, path_join("data", "starting_grid.json"));

    RandomForest forest;

    if (!forest.load_model(model_path))
    {
        return 1;
    }

    std::ifstream grid_file(grid_path);
    if (!grid_file.is_open())
    {
        std::cerr << "Error: Could not find starting_grid.json at " << grid_path << "\n";
        return 1;
    }

    json grid_json;
    grid_file >> grid_json;
    std::vector<Driver> grid;

    for (const auto &d : grid_json)
    {
        Driver driver;
        driver.name = d["driver"];
        driver.features = {d["grid_pos"], d["momentum_score"], d["racecraft_rating"]};
        driver.predicted_finish = forest.predict(driver.features);
        grid.push_back(driver);
    }

    std::sort(grid.begin(), grid.end(), compareDrivers);

    std::cout << "\n========================================\n";
    std::cout << "  GRAND PRIX RACE CLASSIFICATION\n";
    std::cout << "========================================\n";

    for (size_t i = 0; i < grid.size(); ++i)
    {
        std::cout << "P" << std::setw(2) << std::left << (i + 1) << " | "
                  << std::setw(4) << grid[i].name
                  << " | (AI Score: " << std::fixed << std::setprecision(2) << grid[i].predicted_finish << ")\n";
    }
    std::cout << "========================================\n";

    return 0;
}