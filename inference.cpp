#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "include/nlohmann/json.hpp"

using json = nlohmann::json;

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

// Custom sorting function for the leaderboard
bool compareDrivers(const Driver &a, const Driver &b)
{
    return a.predicted_finish < b.predicted_finish;
}

int main()
{
    RandomForest forest;

    if (!forest.load_model("model_metadata.json"))
    {
        return 1;
    }

    // Load the Weekend Grid
    std::ifstream grid_file("starting_grid.json");
    if (!grid_file.is_open())
    {
        std::cerr << "Error: Could not find starting_grid.json\n";
        return 1;
    }

    json grid_json;
    grid_file >> grid_json;
    std::vector<Driver> grid;

    // Run AI prediction for each driver
    for (const auto &d : grid_json)
    {
        Driver driver;
        driver.name = d["driver"];
        driver.features = {d["grid_pos"], d["recent_form"]};
        driver.predicted_finish = forest.predict(driver.features);
        grid.push_back(driver);
    }

    // Sort grid from 1st place to Last place
    std::sort(grid.begin(), grid.end(), compareDrivers);

    // Output final results
    std::cout << "\n========================================\n";
    std::cout << "🏁 GRAND PRIX RACE CLASSIFICATION 🏁\n";
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