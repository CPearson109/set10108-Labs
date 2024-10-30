#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>

std::vector<char> read_file(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    file.close();
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

int calc_token_occurrences(const std::vector<char>& data, const std::string& token)
{
    int numOccurrences = 0;
    int tokenLen = int(token.length());
    for (int i = 0; i <= int(data.size() - tokenLen); ++i)
    {
        if (strncmp(&data[i], token.c_str(), tokenLen) == 0) {
            auto iPrefix = i - 1;
            auto iSuffix = i + tokenLen;

            if ((iPrefix < 0 || !isalpha(data[iPrefix])) &&
                (iSuffix >= int(data.size()) || !isalpha(data[iSuffix])))
            {
                ++numOccurrences;
            }
        }
    }
    return numOccurrences;
}

int main()
{
    std::string datasetFolder = "dataset";
    std::vector<std::string> txtFiles;

    std::cout << "Available text files in '" << datasetFolder << "':\n";
    for (const auto& entry : std::filesystem::directory_iterator(datasetFolder)) {
        if (entry.path().extension() == ".txt") {
            txtFiles.push_back(entry.path().filename().string());
            std::cout << " - " << entry.path().filename().string() << std::endl;
        }
    }

    std::string filename;
    std::cout << "Enter the filename you want to search in: ";
    std::cin >> filename;
    std::string filepath = datasetFolder + "/" + filename;

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    std::string inputWords;
    std::cout << "Enter words to search for, separated by a comma (','): ";
    std::cin.ignore();
    std::getline(std::cin, inputWords);

    std::istringstream iss(inputWords);
    std::string word;
    while (std::getline(iss, word, ',')) {
        word.erase(std::remove(word.begin(), word.end(), ' '), word.end()); // Remove any whitespace
        auto start = std::chrono::high_resolution_clock::now();
        int occurrences = calc_token_occurrences(file_data, word);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Word: " << word << "\n"
            << "Occurrences: " << occurrences << "\n"
            << "Time taken: " << elapsed << " ms\n\n";
    }

    return 0;
}
