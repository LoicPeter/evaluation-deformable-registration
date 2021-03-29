// -------------------------------------------------------------------
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
// -------------------------------------------------------------------


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
