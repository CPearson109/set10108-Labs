#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>

std::vector<char> read_file(const char* filename)
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

int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i < int(data.size()); ++i)
    {
        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;
        ++numOccurrences;
    }
    return numOccurrences;
}

int main()
{
    const char* filepath = "dataset/shakespeare.txt";
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    const char* words[] = { "the"};
    for (const char* word : words)
    {
        int totalOccurrences = 0;
        double totalTime = 0.0;
        double fastestTime = std::numeric_limits<double>::max();
        double slowestTime = 0.0;
        int runCount = 1000;

        for (int run = 0; run < runCount; ++run)
        {
            auto start = std::chrono::high_resolution_clock::now();
            int occurrences = calc_token_occurrences(file_data, word);
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            totalOccurrences = occurrences;
            totalTime += elapsed;

            if (elapsed < fastestTime)
                fastestTime = elapsed;
            if (elapsed > slowestTime)
                slowestTime = elapsed;
        }

        double averageTime = totalTime / runCount;
        std::cout << "Word: " << word << "\n"
            << "Occurrences: " << totalOccurrences << "\n"
            << "Fastest time: " << fastestTime << " ms\n"
            << "Slowest time: " << slowestTime << " ms\n"
            << "Average time: " << averageTime << " ms\n\n";
    }

    return 0;
}
