#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
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
    std::string model_type;
    double learning_rate;
    double init_value;

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

        // Read model type (defaults to random_forest for backward compatibility)
        model_type = j.value("model_type", "random_forest");
        learning_rate = j.value("learning_rate", 0.1);
        init_value = j.value("init_value", 0.0);

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

        if (model_type == "gradient_boosting")
        {
            // Gradient Boosting: init_value + learning_rate * sum(tree predictions)
            double total_score = init_value;
            for (const auto &tree : trees)
            {
                total_score += learning_rate * tree.predict(features);
            }
            return total_score;
        }
        else
        {
            // Random Forest: average of all tree predictions
            double total_score = 0.0;
            for (const auto &tree : trees)
            {
                total_score += tree.predict(features);
            }
            return total_score / trees.size();
        }
    }
};

struct Driver
{
    std::string name;
    int grid_pos;
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
    std::string predictions_path = path_join(project_root, path_join("data", "predictions.json"));

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
        driver.grid_pos = static_cast<int>(d["GridPosition"].get<double>());

        // Read all 5 features: GridPosition, Momentum_Score, Racecraft_Rating,
        //                       Constructor_Strength, Consistency
        driver.features = {
            d["GridPosition"].get<double>(),
            d.value("Momentum_Score", 0.0),
            d.value("Racecraft_Rating", 0.0),
            d.value("Constructor_Strength", 0.0),
            d.value("Consistency", 2.89)
        };

        driver.predicted_finish = forest.predict(driver.features);
        grid.push_back(driver);
    }

    std::sort(grid.begin(), grid.end(), compareDrivers);

    // --- Display Race Classification ---
    double leader_score = grid.empty() ? 0.0 : grid[0].predicted_finish;

    std::cout << "\n==========================================\n";
    std::cout << "   GRAND PRIX RACE CLASSIFICATION\n";
    std::cout << "==========================================\n";
    std::cout << std::left << std::setw(5) << "POS"
              << std::setw(5) << "DRV"
              << std::setw(6) << "GRID"
              << std::setw(10) << "SCORE"
              << "GAP\n";
    std::cout << "------------------------------------------\n";

    for (size_t i = 0; i < grid.size(); ++i)
    {
        double gap = grid[i].predicted_finish - leader_score;
        std::cout << "P" << std::setw(3) << std::left << (i + 1) << " "
                  << std::setw(5) << grid[i].name
                  << std::setw(6) << grid[i].grid_pos
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << grid[i].predicted_finish;

        if (i == 0)
        {
            std::cout << "LEADER";
        }
        else
        {
            std::cout << "+" << std::fixed << std::setprecision(2) << gap;
        }
        std::cout << "\n";
    }
    std::cout << "==========================================\n";

    // --- Save predictions JSON ---
    json predictions = json::array();
    for (size_t i = 0; i < grid.size(); ++i)
    {
        json pred;
        pred["driver"] = grid[i].name;
        pred["grid_pos"] = grid[i].grid_pos;
        pred["position"] = static_cast<int>(i + 1);
        pred["score"] = std::round(grid[i].predicted_finish * 100.0) / 100.0;
        predictions.push_back(pred);
    }

    std::ofstream out_file(predictions_path);
    if (out_file.is_open())
    {
        out_file << predictions.dump(2) << "\n";
        out_file.close();
    }

    return 0;
}