#include "files.h"
#include "timer.h"

//the first step is to read in the MINST data set
// Each file has 1000 training examples. Each training example is of size 28x28 pixels. 
// The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. 
// The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.
int main(){

}
//this sudo code is for each image being a 1d array
// for (blurrow = -BLUR_SIZE to BLUR_SIZE){
//     for (blurcol = -BLUR_SIZE to BLUR_SIZE){
//     currow = row + blurrow;
//     curcol = col + blurcol;
//     // take care of the image edge and ensure valid image pixel
//     if (currow > -1 && currow < height && curcol > -1 && curcol < width){
//         pixVal += inputImage[currow * width + curcol];
//         pixels++; // number of pixels added
//         }
//     }
// }
// // Write the new pixel value in outImage
// outImage[row * width + col] = (variable type)(pixVal / pixels);
