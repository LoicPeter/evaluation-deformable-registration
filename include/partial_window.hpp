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


#ifndef PARTIAL_WINDOW_H_INCLUDED
#define PARTIAL_WINDOW_H_INCLUDED

#include <string>
#include <opencv2/core/core.hpp>

class PartialWindow
{
    public:
    PartialWindow(const std::string& partial_window_name, int partial_window_width, int partial_window_height, const cv::Mat& input_image, int initial_x0 = 0, int initial_y0 = 0, int window_position_x = -1,  int window_position_y = -1);
    PartialWindow(const std::string& partial_window_name, int partial_window_width, int partial_window_height, int initial_x0 = 0, int initial_y0 = 0,  int window_position_x = -1,  int window_position_y = -1);
    ~PartialWindow();
    void centre_on_point(const cv::Point2f& input_point, const cv::Mat& input_image) const;
    void reset(const cv::Mat& input_image, int initial_x0 = 0, int initial_y0 = 0);
    
    
    std::string name;
    int width;
    int height;
    int default_width;
    int default_height;
    int default_window_position_x;  
    int default_window_position_y;
    int *current_origin;
    
    private:
    void adjust_dimensions_and_create_trackbar(int dimx, int dimy);
    

};


void imshow(const PartialWindow& partial_window, const cv::Mat& input_image);


#endif
