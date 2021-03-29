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


#include "partial_window.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

PartialWindow::PartialWindow(const std::string& partial_window_name, int partial_window_width, int partial_window_height, const cv::Mat& input_image, int initial_x0, int initial_y0, int window_position_x,  int window_position_y)
{
    int dimx = input_image.cols;
    int dimy = input_image.rows;
    
    this->name = partial_window_name;
    
    this->current_origin = new int[2];
    (this->current_origin)[0] = initial_x0;
    (this->current_origin)[1] = initial_y0;
    
    this->default_width = partial_window_width;
    this->default_height = partial_window_height;
    this->default_window_position_x = window_position_x;
    this->default_window_position_y = window_position_y;
    
    cv::namedWindow(partial_window_name,CV_WINDOW_AUTOSIZE); // or CV_WINDOW_NORMAL ?
    
    this->adjust_dimensions_and_create_trackbar(dimx,dimy);
    
    cv::resizeWindow(partial_window_name,this->width,this->height);
    
    if ((this->default_window_position_x>=0) && (this->default_window_position_y>=0))
        cv::moveWindow(this->name,this->default_window_position_x,this->default_window_position_y);
}

void PartialWindow::adjust_dimensions_and_create_trackbar(int dimx, int dimy)
{
    
    if ((this->default_width)>dimx)
    {
        this->width = dimx;
        (this->current_origin)[0] = 0;
    }
    else
    {
        this->width = this->default_width;
        cv::createTrackbar("X",this->name,this->current_origin,dimx - this->width);
    }
    
    if ((this->default_height)>dimy)
    {
        this->height = dimy;
        (this->current_origin)[1]= 0;
    }
    else
    {
        this->height = this->default_height;
        cv::createTrackbar("Y",this->name,this->current_origin + 1,dimy - this->height);
    }

}


PartialWindow::PartialWindow(const std::string& partial_window_name, int partial_window_width, int partial_window_height, int initial_x0, int initial_y0,  int window_position_x, int window_position_y)
{
    
    this->name = partial_window_name;
    
    this->current_origin = new int[2];
    (this->current_origin)[0] = initial_x0;
    (this->current_origin)[1] = initial_y0;
    
    this->default_width = partial_window_width;
    this->default_height = partial_window_height;
    this->default_window_position_x = window_position_x;
    this->default_window_position_y = window_position_y;
    
    cv::namedWindow(partial_window_name,CV_WINDOW_AUTOSIZE);
    
    cv::resizeWindow(partial_window_name,this->width,this->height);
    
    if ((this->default_window_position_x>=0) && (this->default_window_position_y>=0))
        cv::moveWindow(this->name,this->default_window_position_x,this->default_window_position_y);
}

void PartialWindow::reset(const cv::Mat& input_image, int initial_x0, int initial_y0)
{
     cv::destroyWindow(this->name);
     delete[] (this->current_origin);
     
     this->current_origin = new int[2];
     (this->current_origin)[0] = initial_x0;
     (this->current_origin)[1] = initial_y0;
     cv::namedWindow(this->name,CV_WINDOW_AUTOSIZE);
     this->adjust_dimensions_and_create_trackbar(input_image.cols,input_image.rows);
     cv::resizeWindow(this->name,this->width,this->height);
     if ((this->default_window_position_x>=0) && (this->default_window_position_y>=0))
        cv::moveWindow(this->name,this->default_window_position_x,this->default_window_position_y);
}

PartialWindow::~PartialWindow()
{
     cv::destroyWindow(this->name);
     delete[] (this->current_origin);
}

void PartialWindow::centre_on_point(const cv::Point2f& input_point, const cv::Mat& input_image) const
{
    int dimx = input_image.cols;
    int dimy = input_image.rows;
    std::cout << input_point.x << " " << input_point.y << std::endl;
    int *c_o = this->current_origin;
    c_o[0] = std::max<int>(0,((int)input_point.x) - (this->width)/2 - 1);
    c_o[0] = std::min<int>(c_o[0],dimx-width);
    c_o[1] = std::max<int>(0,((int)input_point.y) - (this->height)/2 - 1);
    c_o[1] = std::min<int>(c_o[1],dimy-height);
    std::cout << c_o[0] << " " << c_o[1] << std::endl;
    cv::setTrackbarPos("X",this->name,c_o[0]);
    cv::setTrackbarPos("Y",this->name,c_o[1]);
}

void imshow(const PartialWindow& partial_window, const cv::Mat& input_image)
{
    cv::Mat cropped_image = input_image(cv::Rect(partial_window.current_origin[0],partial_window.current_origin[1],partial_window.width,partial_window.height));
    cv::imshow(partial_window.name,cropped_image);
    if ((partial_window.default_window_position_x>=0) && (partial_window.default_window_position_y>=0))
        cv::moveWindow(partial_window.name,partial_window.default_window_position_x,partial_window.default_window_position_y);
}

