/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "gcgraph.hpp"
#include <iostream>
#include <fstream>
#include <limits>
#include <stdlib.h>

using namespace cv;

int centreX = 0;
int centreY = 0;

static void put1dArrayintoFile(std::string fileName, double * matrix, int size)
{
    std::ofstream out((fileName + ".txt").c_str());
    if(!out)
    {  
        std::cout<<"Cannot open output file\n";
        return ;
    }
    for(int x = 0; x < size; x++)
    {
        out << matrix[x]<< " ";
    }
    out.close();
}

static void put2dArrayintoFile(std::string fileName, Mat matrix, int height, int width)
{
    std::ofstream out((fileName + ".txt").c_str());
    if(!out)
    {  
        std::cout<<"Cannot open output file\n";
        return ;
    }
    Point p;
    for( p.y = 0; p.y < height; p.y++ )
    {
        for( p.x = 0; p.x < width; p.x++ )
        {
            
            out << matrix.at<int>(p) << " ";
        }
        out << "\n";
    }
    
    out.close();
}

static void put2dArrayintoFile1(std::string fileName, Mat matrix)
{
    std::ofstream out((fileName + ".txt").c_str());
    if(!out)
    {  
        std::cout<<"Cannot open output file\n";
        return ;
    }

    Point p;
    for( p.y = 0; p.y < matrix.rows; p.y++ )
    {
        for( p.x = 0; p.x < matrix.cols; p.x++ )
        {
            if(matrix.at<uchar>(p) == GC_PR_BGD || matrix.at<uchar>(p) == GC_BGD)
                out << "1" << " ";
            else
                out << "0" << " ";
        }
        out << "\n";
    }
    
    out.close();
}

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut - Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();


    void calcInverseCovAndDeterm( int ci );
    Mat model;
    int sizeCoeff;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

GMM::GMM( Mat& _model )
{
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;
    sizeCoeff = modelSize*componentsCount;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm( ci );
    totalSampleCount = 0;
}

double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n/totalSampleCount;

            double* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

            double* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm =
              covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equal "
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
  Initialize mask using rectangular.
*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );

    centreX = rect.x + (rect.width / 2);
    centreY = rect.y + (rect.height / 2);
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();

    put1dArrayintoFile("bgdGMM_coeff initgmm", bgdGMM.coefs, bgdGMM.sizeCoeff);
    put1dArrayintoFile("bgdGMM_mean initgmm", bgdGMM.mean, bgdGMM.sizeCoeff);
    put1dArrayintoFile("bgdGMM_cov initgmm", bgdGMM.cov, bgdGMM.sizeCoeff);

    put1dArrayintoFile("fgdGMM_coeff initgmm", fgdGMM.coefs, fgdGMM.sizeCoeff);
    put1dArrayintoFile("fgdGMM_mean initgmm", fgdGMM.mean, fgdGMM.sizeCoeff);
    put1dArrayintoFile("fgdGMM_cov initgmm", fgdGMM.cov, fgdGMM.sizeCoeff);
}

/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       GCGraph<double>& graph , bool **dummyarray, double color_weight, double terminal_weight, double smoothness_weight, double shape_weight)
{
    int vtxCount = img.cols*img.rows,
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < centreX; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) ) * color_weight;
                toSink = -log( fgdGMM(color) ) * color_weight;
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda * terminal_weight;
            }
            else // GC_FGD
            {
                fromSource = lambda * terminal_weight;
                toSink = 0;
            }
            
            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p) * smoothness_weight;
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p) * smoothness_weight;
                Point p_neighbour;
                p_neighbour.x = p.x - 1;
                p_neighbour.y = p.y - 1;
                if(shape_weight > 0 && dummyarray[p_neighbour.y][p_neighbour.x] == true && p.x < centreX && p.y < centreY && p_neighbour.x < centreX && p_neighbour.y < centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p.y][p.x] = true;
                    // mask.at<uchar>(p) = GC_FGD;
                }
                if(shape_weight > 0 && dummyarray[p.y][p.x] == true && p.x > centreX && p.y > centreY && p_neighbour.x > centreX && p_neighbour.y > centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p_neighbour.y][p_neighbour.x] = true;
                }

                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p) * smoothness_weight;
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p) * smoothness_weight;
                Point p_neighbour;
                p_neighbour.x = p.x + 1;
                p_neighbour.y = p.y - 1;
                if(shape_weight > 0 && dummyarray[p_neighbour.y][p_neighbour.x] == true && p.x > centreX && p.y < centreY && p_neighbour.x > centreX && p_neighbour.y < centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p.y][p.x] = true;
                }
                if(shape_weight > 0 && dummyarray[p.y][p.x] == true && p.x < centreX && p.y > centreY && p_neighbour.x < centreX && p_neighbour.y > centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p_neighbour.y][p_neighbour.x] = true;
                }
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );
        }
    }
    for( p.y = img.rows - 1; p.y >= centreX; p.y-- )
    {
                for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) ) * color_weight;
                toSink = -log( fgdGMM(color) ) * color_weight;
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda * terminal_weight;
            }
            else // GC_FGD
            {
                fromSource = lambda * terminal_weight;
                toSink = 0;
            }
            if(mask.at<uchar>(p) == GC_FGD)
            {
                dummyarray[p.y][p.x] = true;
            }
            
            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p) * smoothness_weight;
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p) * smoothness_weight;
                Point p_neighbour;
                p_neighbour.x = p.x - 1;
                p_neighbour.y = p.y - 1;
                if(shape_weight > 0 && dummyarray[p_neighbour.y][p_neighbour.x] == true && p.x < centreX && p.y < centreY && p_neighbour.x < centreX && p_neighbour.y < centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p.y][p.x] = true;
                    // mask.at<uchar>(p) = GC_FGD;
                }
                if(shape_weight > 0 && dummyarray[p.y][p.x] == true && p.x > centreX && p.y > centreY && p_neighbour.x > centreX && p_neighbour.y > centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p_neighbour.y][p_neighbour.x] = true;
                }

                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p) * smoothness_weight;
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p) * smoothness_weight;
                Point p_neighbour;
                p_neighbour.x = p.x + 1;
                p_neighbour.y = p.y - 1;
                if(shape_weight > 0 && dummyarray[p_neighbour.y][p_neighbour.x] == true && p.x > centreX && p.y < centreY && p_neighbour.x > centreX && p_neighbour.y < centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p.y][p.x] = true;
                }
                if(shape_weight > 0 && dummyarray[p.y][p.x] == true && p.x < centreX && p.y > centreY && p_neighbour.x < centreX && p_neighbour.y > centreY)
                {
                    fromSource = shape_weight;
                    toSink = 0;
                    dummyarray[p_neighbour.y][p_neighbour.x] = true;
                }
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );
        }
    }
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask , bool **dummyarray)
{
    graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                {
                    mask.at<uchar>(p) = GC_PR_FGD;
                    dummyarray[p.y][p.x] = true;

                }
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

void cv::grabCut( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode)
{
    CV_INSTRUMENT_REGION()

    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    bool *dummyarray[img.rows];
    for(int i = 0;i < img.rows;i++)
    {
        dummyarray[i] = new bool[img.cols];
        for(int j =0;j<img.cols;j++)
        {
            dummyarray[i][j] = false;
        }
    }

    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image must have CV_8UC3 type" );

    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );

    // put1dArrayintoFile("bgdModelinitial", bgdModel, bgdGMM.sizeCoeff);
    // put1dArrayintoFile("fgdModelinitial", fgdModel, fgdGMM.sizeCoeff);

    Mat compIdxs( img.size(), CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.size(), rect );
        else // flag == GC_INIT_WITH_MASK
            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    if( iterCount <= 0)
        return;

    if( mode == GC_EVAL_FREEZE_MODEL )
        iterCount = 1;

    if( mode == GC_EVAL || mode == GC_EVAL_FREEZE_MODEL )
        checkMask( img, mask );

    const double gamma = 50;
    const double lambda = 9*gamma;
    const double beta = calcBeta( img );

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );
    double color_weight = 0;
    double terminal_weight = 0;
    double smoothness_weight = 0;
    double shape_weight = 0;
    std::string line;
    std::ifstream myfile ("/Users/manikant/Documents/License-Plate-Detection/code/opencv_clone/opencv/weights.txt");
    if (myfile.is_open())
    {
        getline (myfile,line);
        color_weight = atof(line.c_str()) + 0.0;
        getline (myfile,line);
        terminal_weight = atof(line.c_str()) + 0.0;
        getline (myfile,line);
        smoothness_weight = atof(line.c_str()) + 0.0;
        getline (myfile,line);
        shape_weight = atof(line.c_str()) + 0.0;
        myfile.close();
    }
    std::cout<<color_weight <<" "<<terminal_weight << " "<<smoothness_weight <<" "<<shape_weight<< "\n";

    for( int i = 0; i < iterCount; i++ )
    {
        GCGraph<double> graph;
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        put2dArrayintoFile("assigned compInds " + std::to_string(i), compIdxs, img.rows, img.cols);
        // put1dArrayintoFile("bgdModelinitial", bgdModel, bgdGMM.sizeCoeff);
        // put1dArrayintoFile("fgdModelinitial", fgdModel, fgdGMM.sizeCoeff);
        if( mode != GC_EVAL_FREEZE_MODEL )
            learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        put1dArrayintoFile("bgdGMM_coeff learnGMMs"+ std::to_string(i), bgdGMM.coefs, bgdGMM.sizeCoeff);
        put1dArrayintoFile("bgdGMM_mean learnGMMs"+ std::to_string(i), bgdGMM.mean, bgdGMM.sizeCoeff);
        put1dArrayintoFile("bgdGMM_cov learnGMMs"+ std::to_string(i), bgdGMM.cov, bgdGMM.sizeCoeff);

        put1dArrayintoFile("fgdGMM_coeff learnGMMs"+ std::to_string(i), fgdGMM.coefs, fgdGMM.sizeCoeff);
        put1dArrayintoFile("fgdGMM_mean learnGMMs"+ std::to_string(i), fgdGMM.mean, fgdGMM.sizeCoeff);
        put1dArrayintoFile("fgdGMM_cov learnGMMs"+ std::to_string(i), fgdGMM.cov, fgdGMM.sizeCoeff);
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph ,dummyarray, color_weight, terminal_weight, smoothness_weight, shape_weight);
        estimateSegmentation( graph, mask , dummyarray);
        put2dArrayintoFile1("assigned mask " + std::to_string(i), mask);
    }
}
