#include "processing_scripts.hpp"
#include <sstream>
#include <fstream>


void write_elastix_settings_textfiles(const std::string& folder)
{
    std::vector<int> iterations = {500, 1000};
    std::vector<int> number_of_resolutions = {4,5,6,7,8};
   // std::vector<std::string> metrics = {"AdvancedMattesMutualInformation", "NormalizedMutualInformation", "AdvancedNormalizedCorrelation", "AdvancedMeanSquares"};
    std::vector<std::string> metrics = {"AdvancedMattesMutualInformation", "NormalizedMutualInformation", "AdvancedNormalizedCorrelation"};
    int indFile(0);
    for (const std::string& m : metrics)
    {
        for (int res : number_of_resolutions)
        {
            for (int it : iterations)
            {
                std::ostringstream oss_indFile;
                oss_indFile << indFile;
                std::ofstream infile(folder + "/elastixconfig" + oss_indFile.str() + ".txt");
                
                // Copy the base settings
                std::ifstream infile_base(folder + "/elastixbaseconfig.txt");
                std::string line;
                while (true)
                {
                    std::getline(infile_base,line);
                    if (!infile_base.eof())
                        infile << line << std::endl;
                    else
                        break;
                }
                
                // Append the specific settings
                infile << "(Metric \"" << m << "\")" << std::endl;
                infile << "(NumberOfResolutions " << res << ")" << std::endl;
                infile << "(MaximumNumberOfIterations ";
                for (int k=0; k<res; ++k)
                    infile << it << " "; 
                infile << ")" << std::endl;
                ++indFile;
            }
        }
    }
}
