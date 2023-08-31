#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Function to generate random double between 0 and 1
double getRandomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

// Naive Bayes classifier
class NaiveBayes {
public:
    void train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) {
        // Assuming two classes: 0 and 1
        for (size_t i = 0; i < features.size(); ++i) {
            int label = labels[i];
            classCounts[label]++;
            for (size_t j = 0; j < features[i].size(); ++j) {
                featureCounts[label][j] += features[i][j];
            }
        }
    }

    int predict(const std::vector<double>& input) {
        int predictedLabel = -1;
        double maxPosterior = -1.0;

        for (const auto& classCountPair : classCounts) {
            int label = classCountPair.first;
            double prior = static_cast<double>(classCountPair.second) / classCounts.size();
            double likelihood = 1.0;

            for (size_t i = 0; i < input.size(); ++i) {
                double featureProbability = featureCounts[label][i] / classCountPair.second;
                // Assuming Gaussian distribution, you can modify this for different distributions
                likelihood *= exp(-pow(input[i] - featureProbability, 2) / (2 * featureProbability));
            }

            double posterior = prior * likelihood;
            if (posterior > maxPosterior) {
                maxPosterior = posterior;
                predictedLabel = label;
            }
        }

        return predictedLabel;
    }

private:
    std::map<int, int> classCounts;
    std::map<int, std::vector<double>> featureCounts;
};

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    // Generate random training data
    std::vector<std::vector<double>> features = {{getRandomDouble(), getRandomDouble()},
                                                  {getRandomDouble(), getRandomDouble()},
                                                  {getRandomDouble(), getRandomDouble()}};
    std::vector<int> labels = {0, 1, 0};

    // Train Naive Bayes classifier
    NaiveBayes classifier;
    classifier.train(features, labels);

    // Generate a random test input
    std::vector<double> testInput = {getRandomDouble(), getRandomDouble()};

    // Predict the label using the trained classifier
    int predictedLabel = classifier.predict(testInput);

    std::cout << "Test Input: " << testInput[0] << ", " << testInput[1] << std::endl;
    std::cout << "Predicted Label: " << predictedLabel << std::endl;

    return 0;
}
