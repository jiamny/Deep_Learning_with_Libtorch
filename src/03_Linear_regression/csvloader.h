#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>

class CSVRow {
public:
	float operator[](std::size_t index) {
		std::string &eg = m_data[index];
		return std::atof(eg.c_str());
	}
	std::size_t size() const {
		return m_data.size();
	}
	void readNextRow(std::istream &str) {
		std::string line;
		std::getline(str, line);

		std::stringstream lineStream(line);
		std::string cell;

		m_data.clear();
		while (std::getline(lineStream, cell, ',')) {
			m_data.push_back(cell);
		}
		// This checks for a trailing comma with no data after it.
		if (!lineStream && cell.empty()) {
			// If there was a trailing comma then add an empty element.
			m_data.push_back("");
		}
	}
private:
	std::vector<std::string> m_data;
};

std::istream& operator>>(std::istream &str, CSVRow &data) {
	data.readNextRow(str);
	return str;
}

template<typename T>
std::vector<T> linspace(int start, int end, int length) {
	std::vector<T> vec;
	T diff = (end - start) / T(length);
	for (int i = 0; i < length; i++) {
		vec.push_back(start + diff * i);
	}
	return vec;
}

// This function returns random numbers of length: length, and multiplies each by multiplier
template<typename T>
std::vector<T> random(int length, int multiplier) {
	std::vector<T> vec;

	for (int i = 0; i < length; i++) {
		vec.push_back((rand() % 10) * 2.0);
	}

	return vec;
}

// This function adds two vectors and returns the sum vector
template<typename T>
std::vector<T> add_two_vectors(std::vector<T> const &a_vector,
		std::vector<T> const &b_vector) {
	// assert both are of same size
	assert(a_vector.size() == b_vector.size());

	std::vector<T> c_vector;
	std::transform(std::begin(a_vector), std::end(a_vector),
			std::begin(b_vector), std::back_inserter(c_vector),
			[](T const &a, T const &b) {
				return a + b;
			});

	return c_vector;
}

// This creates a data for Linear Regression
template<typename T>
std::pair<std::vector<T>, std::vector<T>> create_data() {
	int64_t m = 4; // Slope
	int64_t c = 6; // Intercept

	int start = 0;
	int end = 11;
	int length = 91;
	std::vector<T> y = linspace<T>(start, end, length);
	std::vector<T> x = y;

	// TODO: assert length of y == length
	// Target: y = mx + c + <something random> 

	// Source: https://stackoverflow.com/a/3885136
	// This multiplies the vector with a scalar
	// y = y * m
	std::transform(y.begin(), y.end(), y.begin(), [m](long long val) {
		return val * m;
	});

	// Source: https://stackoverflow.com/a/4461466
	// y = y + c
	std::transform(y.begin(), y.end(), y.begin(), [c](long long val) {
		return val + c;
	});

	// y = y + <random numbers>
	// There are total 91 numbers
	// y = y + random(91, 2) // calculate 91 random numbers and multiply each by 2
	std::vector<T> random_vector = random<T>(91, 2);
	std::vector<T> vec_sum_y = add_two_vectors<T>(y, random_vector);

	return std::make_pair(x, vec_sum_y);
}

// Normalize Feature, Formula: (x - min)/(max - min)
std::vector<float> normalize_feature(std::vector<float> feat) {
	using ConstIter = std::vector<float>::const_iterator;
	ConstIter max_element; //= *std::max_element(feat.begin(), feat.end());
	ConstIter min_element; //= *std::min_element(feat.begin(), feat.end());
	std::tie(min_element, max_element) = std::minmax_element(std::begin(feat),
			std::end(feat));

	float extra = *max_element == *min_element ? 1.0 : 0.0;
	for (auto &val : feat) {
		// max_element - min_element + 1 to avoid divide by zero error
		val = (val - *min_element) / (*max_element - *min_element + extra);
	}

	return feat;
}

// This function processes data, Loads CSV file to vectors and normalizes features to (0, 1)
// Assumes last column to be label and first row to be header (or name of the features)
std::pair<std::vector<float>, std::vector<float>> process_data(
		std::ifstream &file) {
	std::vector<std::vector<float>> features;
	std::vector<float> label;

	CSVRow row;
	// Read and throw away the first row.
	file >> row;

	while (file >> row) {
		features.emplace_back();
		for (std::size_t loop = 0; loop < row.size(); ++loop) {
			features.back().emplace_back(row[loop]);
		}
		features.back() = normalize_feature(features.back());

		// Push final column to label vector
		label.push_back(row[row.size() - 1]);
	}

	// Flatten features vectors to 1D
	std::vector<float> inputs = features[0];
	int64_t total = std::accumulate(std::begin(features) + 1,
			std::end(features), 0UL,
			[](std::size_t s, std::vector<float> const &v) {
				return s + v.size();
			});

	inputs.reserve(total);
	for (std::size_t i = 1; i < features.size(); i++) {
		inputs.insert(inputs.end(), features[i].begin(), features[i].end());
	}
	return std::make_pair(inputs, label);
}

// This function processes data, Loads CSV file to vectors and normalizes features to (0, 1)
// Assumes last column to be label and first row to be header (or name of the features)
void process_split_data(std::ifstream &file, std::unordered_set<int> train_idx,
		std::vector<float> &train, std::vector<float> &trainLabel,
		std::vector<float> &test, std::vector<float> &testLabel) {

	std::vector<std::vector<float>> trainData;
	std::vector<std::vector<float>> testData;

	CSVRow row;
	// Read and throw away the first row.
	file >> row;

	int j = 0;

	while (file >> row) {

		if (train_idx.count(j)) {
			trainData.emplace_back();
			for (std::size_t loop = 0; loop < row.size(); ++loop) {
				trainData.back().emplace_back(row[loop]);
			}
			trainData.back() = normalize_feature(trainData.back());

			// Push final column to label vector
			trainLabel.push_back(row[row.size() - 1]);
		} else {
			testData.emplace_back();
			for (std::size_t loop = 0; loop < row.size(); ++loop) {
				testData.back().emplace_back(row[loop]);
			}
			testData.back() = normalize_feature(testData.back());

			// Push final column to label vector
			testLabel.push_back(row[row.size() - 1]);
		}
		j++;
	}

	// Flatten features vectors to 1D
	train = trainData[0];
	int64_t total = std::accumulate(std::begin(trainData) + 1,
			std::end(trainData), 0UL,
			[](std::size_t s, std::vector<float> const &v) {
				return s + v.size();
			});

	train.reserve(total);
	for (std::size_t i = 1; i < trainData.size(); i++) {
		train.insert(train.end(), trainData[i].begin(), trainData[i].end());
	}

	test = testData[0];
	total = std::accumulate(std::begin(testData) + 1, std::end(testData), 0UL,
			[](std::size_t s, std::vector<float> const &v) {
				return s + v.size();
			});

	test.reserve(total);
	for (std::size_t i = 1; i < testData.size(); i++) {
		test.insert(test.end(), testData[i].begin(), testData[i].end());
	}
}
