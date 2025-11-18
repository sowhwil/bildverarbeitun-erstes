#include <cstdio>
#include <iostream>
#include "utility.hpp"
#include "depthai/depthai.hpp"

static std::atomic<bool> downscaleColor{true};
static constexpr int fps = 30;
static constexpr auto monoRes = dai::MonoCameraProperties::SensorResolution::THE_720_P;

int main(int argc, char** argv) {
    using namespace std;

    if(argc < 2) {
        cout << "How to use: " << argv[0] << " <mode>" << endl;
        cout << "  1 - Live camera" << endl;
        cout << "  2 - Record videos" << endl;
        cout << "  3 - Playback videos" << endl;
        return 1;
    }

    int mode = atoi(argv[1]);

    if(mode < 1 || mode > 3) {
        cerr << "Invalid mode. Use 1, 2, or 3." << endl;
        return 1;
    }

    cv::VideoWriter rgbWriter, depthWriter;
    cv::VideoCapture rgbCapture, depthCapture;

    if(mode == 3) {
        rgbCapture.open("rgb_video.avi");
        depthCapture.open("depth_video.avi");
        //wird beid durch capture geöffnet

        if(!rgbCapture.isOpened() || !depthCapture.isOpened()) {
            cerr << "Error opening video files" << endl;
            return 1;
        }
        //wenn die einer von den nicht geöffnet werden kann wird abgestürt und warnt error

        cv::namedWindow("rgb");
        cv::namedWindow("depth");
        //ein window mit die beiden name erstellt

        cv::Mat rgbFrame, depthFrame;
        //erstellt ein matrix mit der name

        while(true) {
            rgbCapture >> rgbFrame;
            depthCapture >> depthFrame;
            //wird der geöffneteinhalt in der frame gespeichert und nachher gezeigt

            if(rgbFrame.empty() || depthFrame.empty()) break;
            //falls es empty leer geht der aus der schleife raus

            cv::imshow("rgb", rgbFrame);
            cv::imshow("depth", depthFrame);
            //zeigt die augezeichnete an

            int key = cv::waitKey(30);
            if(key == 'q' || key == 'Q') break;
        }

        rgbCapture.release();
        depthCapture.release();
        cv::destroyAllWindows();
        return 0;
    }

    dai::Pipeline pipeline;
    dai::Device device;
    std::vector<std::string> queueNames;

    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto left = pipeline.create<dai::node::MonoCamera>();
    auto right = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();

    auto rgbOut = pipeline.create<dai::node::XLinkOut>();
    auto depthOut = pipeline.create<dai::node::XLinkOut>();

    rgbOut->setStreamName("rgb");
    queueNames.push_back("rgb");
    depthOut->setStreamName("depth");
    queueNames.push_back("depth");

    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setFps(fps);
    if(downscaleColor) camRgb->setIspScale(2, 3);

    try {
        auto calibData = device.readCalibration2();
        auto lensPosition = calibData.getLensPosition(dai::CameraBoardSocket::CAM_A);
        if(lensPosition) {
            camRgb->initialControl.setManualFocus(lensPosition);
        }
    } catch(const std::exception& ex) {
        cout << ex.what() << endl;
        return 1;
    }

    left->setResolution(monoRes);
    left->setCamera("left");
    left->setFps(fps);
    right->setResolution(monoRes);
    right->setCamera("right");
    right->setFps(fps);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->setLeftRightCheck(true);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    stereo->setExtendedDisparity(true);

    camRgb->isp.link(rgbOut->input);
    left->out.link(stereo->left);
    right->out.link(stereo->right);
    stereo->disparity.link(depthOut->input);

    device.startPipeline(pipeline);

    for(const auto& name : queueNames) {
        device.getOutputQueue(name, 4, false);
    }

    cv::namedWindow("rgb");
    cv::namedWindow("depth");

    if(mode == 2) {
        int frameWidth = downscaleColor ? 1280 : 1920;
        int frameHeight = downscaleColor ? 720 : 1080;

        rgbWriter.open("rgb_video.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(frameWidth, frameHeight));
        depthWriter.open("depth_video.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(frameWidth, frameHeight));

        if(!rgbWriter.isOpened() || !depthWriter.isOpened()) {
            cerr << "Error opening video writers" << endl;
            return 1;
        }

        cout << "Recording started. Press 'q' to stop." << endl;
    }

    while(true) {
        std::unordered_map<std::string, std::shared_ptr<dai::ImgFrame>> latestPacket;

        auto queueEvents = device.getQueueEvents(queueNames);
        for(const auto& name : queueEvents) {
            auto packets = device.getOutputQueue(name)->tryGetAll<dai::ImgFrame>();
            auto count = packets.size();
            if(count > 0) {
                latestPacket[name] = packets[count - 1];
            }
        }

        cv::Mat rgbFrame, depthFrame;

        if(latestPacket.find("rgb") != latestPacket.end()) {
            rgbFrame = latestPacket["rgb"]->getCvFrame();
            cv::imshow("rgb", rgbFrame);
        }

        if(latestPacket.find("depth") != latestPacket.end()) {
            depthFrame = latestPacket["depth"]->getFrame();
            auto maxDisparity = stereo->initialConfig.getMaxDisparity();
            depthFrame.convertTo(depthFrame, CV_8UC1, 255. / maxDisparity);
            cv::applyColorMap(depthFrame, depthFrame, cv::COLORMAP_JET);
            cv::imshow("depth", depthFrame);
        }

        if(mode == 2 && !rgbFrame.empty() && !depthFrame.empty()) {
            rgbWriter.write(rgbFrame);
            depthWriter.write(depthFrame);
        }

        int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q') {
            break;
        }
    }

    if(mode == 2) {
        rgbWriter.release();
        depthWriter.release();
        cout << "Recording stopped." << endl;
    }

    cv::destroyAllWindows();
    return 0;
}
