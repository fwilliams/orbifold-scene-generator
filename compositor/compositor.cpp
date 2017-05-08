#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfChannelList.h>

using namespace std;
using namespace Imf;
using namespace Imath;

struct Options {
    unsigned m_w = 512, m_h = 512;
    std::string m_dirname = "";
    std::string m_outname = "result.exr";
};

Array2D<float> rB, gB, bB, dB, rIB, gIB, bIB;
Options opts;


bool parse_options(int argc, char** argv, Options& outOptions) {
    namespace po = boost::program_options;

    outOptions.m_w = 0;
    outOptions.m_h = 0;
    outOptions.m_dirname = "";
    outOptions.m_outname = "";

    po::options_description desc("Usage: compositor image-directory [options]");
    desc.add_options()
            ("help", "print help message")
            ("image-width,w", po::value<unsigned>(), "The width, in pixels, of the output image. The default is 800.")
            ("image-height,h", po::value<unsigned>(), "The height, in pixels, of the output image. The default is 600.")
            ("out-image-name,n", po::value<std::string>(), "The name of the output image. The default is screenshot.tga.")
            ("image-directory", po::value<std::string>(), "The directory containing the input images.");
    po::positional_options_description po_desc;
    po_desc.add("image-directory", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(po_desc).run(), vm);
    po::notify(vm);

    if(vm.count("help")) {
        std::cout << desc << std::endl;
        return false;
    }

    if(vm.count("image-width")) {
        outOptions.m_w = vm["image-width"].as<unsigned>();
    }
    if(vm.count("image-height")) {
        outOptions.m_h = vm["image-height"].as<unsigned>();
    }

    if(vm.count("out-image-name")) {
        outOptions.m_outname = vm["out-image-name"].as<std::string>();
    }

    if(vm.count("image-directory")) {
        outOptions.m_dirname = vm["image-directory"].as<std::string>();
    } else {
        std::cerr << "Error: image directory not specified." << std::endl;
        std::cout << desc << std::endl;
        return false;
    }

    return true;
}


void read_exr_rgb(const char* filename, Array2D<float>& r, Array2D<float>& g, Array2D<float>& b, int& width, int& height) {
    InputFile file(filename);
    Box2i dw = file.header().dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;

    r.resizeErase(height, width);
    g.resizeErase(height, width);
    b.resizeErase(height, width);

    FrameBuffer frameBuffer;
    frameBuffer.insert("R",
                       Slice(FLOAT, (char*)(&r[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(r[0][0])*1, sizeof(r[0][0])*width, 1, 1, 0.0));
    frameBuffer.insert("G",
                       Slice(FLOAT, (char*)(&g[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(g[0][0])*1, sizeof(g[0][0])*width, 1, 1, 0.0));
    frameBuffer.insert("B",
                       Slice(FLOAT, (char*)(&b[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(b[0][0])*1, sizeof(b[0][0])*width, 1, 1, 0.0));
    file.setFrameBuffer(frameBuffer);
    file.readPixels(dw.min.y, dw.max.y);
}


void read_exr_rgbd(const char* filename, Array2D<float>& r, Array2D<float>& g, Array2D<float>& b, Array2D<float>& d, int& width, int& height) {
    InputFile file(filename);
    Box2i dw = file.header().dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;

    r.resizeErase(height, width);
    g.resizeErase(height, width);
    b.resizeErase(height, width);
    d.resizeErase(height, width);

    FrameBuffer frameBuffer;
    frameBuffer.insert("color.R",
                       Slice(FLOAT, (char*)(&r[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(r[0][0])*1, sizeof(r[0][0])*width, 1, 1, 0.0));
    frameBuffer.insert("color.G",
                       Slice(FLOAT, (char*)(&g[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(g[0][0])*1, sizeof(g[0][0])*width, 1, 1, 0.0));
    frameBuffer.insert("color.B",
                       Slice(FLOAT, (char*)(&b[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(b[0][0])*1, sizeof(b[0][0])*width, 1, 1, 0.0));
    frameBuffer.insert("distance.Y",
                       Slice(FLOAT, (char*)(&d[0][0] - dw.min.x - dw.min.y * width),
                             sizeof(d[0][0])*1, sizeof(d[0][0])*width, 1, 1, FLT_MAX));
    file.setFrameBuffer(frameBuffer);
    file.readPixels(dw.min.y, dw.max.y);
}


void print_exr_channels(const char* filename) {
    InputFile file(filename);

    cout << "File: " << filename << ":" << endl;
    const ChannelList &channels = file.header().channels();
    for(ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
        const Channel& channel = i.channel();
        string type_str;
        switch(channel.type) {
        case PixelType::FLOAT:
            type_str = string("float");
            break;
        case PixelType::HALF:
            type_str = string("half");
            break;
        case PixelType::UINT:
            type_str = string("uint");
            break;
        default:
            assert(false);
            type_str = string("wtf");
            break;
        }
        cout << "  " << i.name() << " type = " << type_str << endl;
    }
}


void write_exr_channels(const char* filename, Array2D<float>& r, Array2D<float>& g, Array2D<float>& b, int width, int height) {
    Header header(width, height);
    Box2i dw = header.dataWindow();

    header.channels().insert("R", Channel(FLOAT));
    header.channels().insert("G", Channel(FLOAT));
    header.channels().insert("B", Channel(FLOAT));
    OutputFile file(filename, header);
    FrameBuffer frameBuffer;

    char* rbase = (char*)(&r[0][0] - dw.min.x - dw.min.y * width);
    char* gbase = (char*)(&g[0][0] - dw.min.x - dw.min.y * width);
    char* bbase = (char*)(&b[0][0] - dw.min.x - dw.min.y * width);

    frameBuffer.insert("R", Slice(FLOAT, rbase, sizeof(float) * 1, sizeof(float) * width));
    frameBuffer.insert("G", Slice(FLOAT, gbase, sizeof(float) * 1, sizeof(float) * width));
    frameBuffer.insert("B", Slice(FLOAT, bbase, sizeof(float) * 1, sizeof(float) * width));
    file.setFrameBuffer(frameBuffer);
    file.writePixels(height);
}

void update_image(size_t w, size_t h,  Array2D<float>& rC, Array2D<float>& gC, Array2D<float>& bC, Array2D<float>& rI, Array2D<float>& gI, Array2D<float>& bI, Array2D<float>& d) {

    cout << "depth[0][0] = " << d[0][0] << endl;

    // This is the first update, just copy data over.
    if (dB.width() == 0 && dB.height() == 0) {
        rB.resizeErase(opts.m_h, opts.m_w);
        gB.resizeErase(opts.m_h, opts.m_w);
        bB.resizeErase(opts.m_h, opts.m_w);
        dB.resizeErase(opts.m_h, opts.m_w);
        rIB.resizeErase(opts.m_h, opts.m_w);
        gIB.resizeErase(opts.m_h, opts.m_w);
        bIB.resizeErase(opts.m_h, opts.m_w);

        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                rB[y][x] = rC[y][x];
                gB[y][x] = gC[y][x];
                bB[y][x] = bC[y][x];
                rIB[y][x] = rI[y][x];
                gIB[y][x] = gI[y][x];
                bIB[y][x] = bI[y][x];
                if (d[y][x] < 0.0) {
                    dB[y][x] = FLT_MAX;
                } else {
                    dB[y][x] = d[y][x];
                }
            }
        }
        return;
    }

    const float DEPTH_EQUAL_THRESH = 10.0;

    for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
            const float curDepth = dB[y][x];
            float cmpDepth = d[y][x];
//            const float curIncR = rIB[y][x];
//            const float curIncG = gIB[y][x];
//            const float curIncB = bIB[y][x];

            if (cmpDepth < 0.0) {
                cmpDepth = FLT_MAX;
            }
            if (fabs(curDepth - cmpDepth) < DEPTH_EQUAL_THRESH) {
                if (rIB[y][x] > rI[y][x]) {
                    rB[y][x] = rC[y][x];
                    rIB[y][x] = rI[y][x];
                }
                if (gIB[y][x] > gI[y][x]) {
                    gB[y][x] = gC[y][x];
                    gIB[y][x] = gI[y][x];
                }
                if (bIB[y][x] > bI[y][x]) {
                    bB[y][x] = bC[y][x];
                    bIB[y][x] = bI[y][x];
                }

//                rB[y][x] = (rB[y][x]*curIncR + rC[y][x]*rI[y][x]) / (curIncR + rI[y][x]);
//                gB[y][x] = (gB[y][x]*curIncG + gC[y][x]*gI[y][x]) / (curIncG + gI[y][x]);
//                bB[y][x] = (bB[y][x]*curIncB + bC[y][x]*bI[y][x]) / (curIncB + bI[y][x]);
//                rIB[y][x] = min(curIncR, rI[y][x]);
//                gIB[y][x] = min(curIncG, gI[y][x]);
//                bIB[y][x] = min(curIncB, bI[y][x]);
//                rB[y][x] = rC[y][x];
//                gB[y][x] = gC[y][x];
//                bB[y][x] = bC[y][x];
//                rIB[y][x] = rI[y][x];
//                gIB[y][x] = gI[y][x];
//                bIB[y][x] = bI[y][x];
            } else if (curDepth < cmpDepth) { // The new pixel is behind this pixel. Disregard it.
                continue;
            } else if (curDepth > cmpDepth) { // The new pixel is in front of this pixel. Choose it.
                rB[y][x] = rC[y][x];
                gB[y][x] = gC[y][x];
                bB[y][x] = bC[y][x];
                rIB[y][x] = rI[y][x];
                gIB[y][x] = gI[y][x];
                bIB[y][x] = bI[y][x];
                dB[y][x] = d[y][x];
            }
        }
    }
}


int main(int argc, char** argv) {
    namespace fs = boost::filesystem;
    using namespace std;

    if (!parse_options(argc, argv, opts)) {
        return 1;
    }

    // Load each layer from the image directory and composite it into the result
    for(auto entry : boost::make_iterator_range(fs::directory_iterator(fs::path(opts.m_dirname)))) {
        const std::string filename = entry.path().filename().string();

        int w, h;
        Array2D<float> rC, gC, bC, rI, gI, bI, d;

        if(!boost::starts_with(filename, "inc_") && boost::ends_with(filename, ".exr")) {
            string color_filename = opts.m_dirname + string("/") + filename;
            string inc_filename = opts.m_dirname + string("/inc_") + filename;

            print_exr_channels(color_filename.c_str());
            read_exr_rgb(color_filename.c_str(), rC, gC, bC, w, h);
            assert(w == opts.m_w && h == opts.m_h);

            print_exr_channels(inc_filename.c_str());
            read_exr_rgbd(inc_filename.c_str(), rI, gI, bI, d, w, h);
            assert(w == opts.m_w && h == opts.m_h);


            update_image(opts.m_w, opts.m_h, rC, gC, bC, rI, gI, bI, d);
//            write_exr_channels("output.exr", rC, gC, bC, opts.m_w, opts.m_h);
        }
    }

    cout << FLT_MAX << endl;
    cout << "Writing output file " << opts.m_outname << endl;
    write_exr_channels("output.exr", rB, gB, bB, opts.m_w, opts.m_h);
}
