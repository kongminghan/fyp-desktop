using Emgu.CV;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Diagnostics;
using Firebase.Database;
using Firebase.Database.Query;
using Test.Model;
using Microsoft.ProjectOxford.Vision.Contract;
using Microsoft.ProjectOxford.Vision;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading;
using Firebase.Storage;
using Emgu.CV.VideoSurveillance;
using System.Drawing.Imaging;
using System.Drawing;
using Google.Apis.Auth.OAuth2;
using Google.Apis.Vision.v1;
using Google.Apis.Http;
using Google.Apis.Services;
using Google.Apis.Vision.v1.Data;

namespace Test
{
    public partial class Form1 : Form
    {
        //private LicensePlateDetector _licensePlateDetector;
        private VideoCapture _capture = null;
        private Mat _frame;
        private MotionHistory _motionHistory;
        private BackgroundSubtractor _forgroundDetector;
        bool ready = true;
        bool found = false;
        Task myTask;
        GoogleCredential credentails;
        VisionService service;

        OpenFileDialog openFileDialog = new OpenFileDialog();

        public Form1()
        {
            InitializeComponent();
            CvInvoke.UseOpenCL = false;

            //create service
            credentails = CreateCredentials("service_account.json");
            service = CreateService("Park", credentails);

            myTask = Task.Run(() =>{});

            try
            {
                _capture = new VideoCapture();
            }
            catch (NullReferenceException excpt)
            {
                MessageBox.Show(excpt.Message);
            }

            _frame = new Mat();
            if (_capture != null && _capture.Ptr != IntPtr.Zero)
            {
                _motionHistory = new MotionHistory(
                    1.0, //in second, the duration of motion history you wants to keep
                    0.05, //in second, maxDelta for cvCalcMotionGradient
                    0.5); //in second, minDelta for cvCalcMotionGradient
                _capture.ImageGrabbed += ProcessFrame;
                _capture.SetCaptureProperty(CapProp.FrameWidth, 650);
                _capture.SetCaptureProperty(CapProp.FrameHeight, 480);
                _capture.SetCaptureProperty(CapProp.Fps, 30);
                _capture.Start();
            }
        }


        private Mat _segMask = new Mat();
        private Mat _forgroundMask = new Mat();
        private void ProcessFrame(object sender, EventArgs arg)
        {
            Mat image = new Mat();
            _capture.Retrieve(image);

            if (_forgroundDetector == null)
            {
                _forgroundDetector = new BackgroundSubtractorMOG2();
            }

            _forgroundDetector.Apply(image, _forgroundMask);
            _motionHistory.Update(_forgroundMask);

            double[] minValues, maxValues;
            Point[] minLoc, maxLoc;
            Mat motionMask = new Mat();

            _motionHistory.Mask.MinMax(out minValues, out maxValues, out minLoc, out maxLoc);
            using (ScalarArray sa = new ScalarArray(255.0 / maxValues[0]))
                CvInvoke.Multiply(_motionHistory.Mask, sa, motionMask, 1, DepthType.Cv8U);

            Mat motionImage = new Mat(motionMask.Size.Height, motionMask.Size.Width, DepthType.Cv8U, 3);
            motionImage.SetTo(new MCvScalar(0));

            double minArea = 1500;
            System.Drawing.Rectangle[] rects;
            using (VectorOfRect boundingRect = new VectorOfRect())
            {
                _motionHistory.GetMotionComponents(_segMask, boundingRect);
                rects = boundingRect.ToArray();
            }

            found = false;

            foreach (System.Drawing.Rectangle comp in rects)
            {
                int area = comp.Width * comp.Height;
                //reject the components that have small area;
                if (area < minArea) continue;

                // find the angle and motion pixel count of the specific area
                double angle, motionPixelCount;
                _motionHistory.MotionInfo(_forgroundMask, comp, out angle, out motionPixelCount);

                //reject the area that contains too few motion
                if (motionPixelCount < area * 160) continue;
                if (!ready) continue;
                found = true;
            }

            if (this.Disposing || this.IsDisposed)
                return;

            imageBox1.Image = image;

            if (ready && found)
            {
                found = false;
                ready = false;
                //_capture.Pause();
                Console.WriteLine("found");
                Stopwatch watch = new Stopwatch();
                watch.Start();
                string filename = DateTime.Now.Millisecond.ToString() + ".jpg";
                image.Save(filename);
                var task = AnnotateAsync(service, filename, "TEXT_DETECTION");
                var result = task.Result;
                if (result != null)
                {
                    OutputToConsole(result, filename, watch);
                }
            }
        }

        public GoogleCredential CreateCredentials(string path)
        {
            GoogleCredential credential;
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read))
            {
                var c = GoogleCredential.FromStream(stream);
                credential = c.CreateScoped(VisionService.Scope.CloudPlatform);
            }

            return credential;
        }

        public VisionService CreateService(string applicationName, IConfigurableHttpClientInitializer credentials)
        {
            var service = new VisionService(new BaseClientService.Initializer()
            {
                ApplicationName = applicationName,
                HttpClientInitializer = credentials
            });

            return service;
        }

        /// Creates the annotation image request.
        private AnnotateImageRequest CreateAnnotationImageRequest(string path, string featureTypes)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Not found.", path);
            }

            var request = new AnnotateImageRequest();
            request.Image = new Google.Apis.Vision.v1.Data.Image();

            var bytes = File.ReadAllBytes(path);
            request.Image.Content = Convert.ToBase64String(bytes);

            request.Features = new List<Feature>();

            request.Features.Add(new Feature() { Type = featureTypes });
           

            return request;
        }

        public async Task<AnnotateImageResponse> AnnotateAsync(VisionService service, string file, string features)
        {
            var request = new BatchAnnotateImagesRequest();
            request.Requests = new List<AnnotateImageRequest>();
            request.Requests.Add(CreateAnnotationImageRequest(file, features));

            var result = await service.Images.Annotate(request).ExecuteAsync();

            if (result?.Responses?.Count > 0)
            {
                return result.Responses[0];
            }

            return null;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Mat gray = new Mat();
            Mat canny = new Mat();

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                //textBox1.Text = openFileDialog.FileName;
                //Mat m = new Mat(openFileDialog.FileName);
                //UMat um = m.GetUMat(AccessType.ReadWrite);
                //imageBox1.Image = um;

                //CvInvoke.CvtColor(um, gray, ColorConversion.Bgr2Gray);
                //CvInvoke.PyrDown(gray, gray);
                //imageBox2.Image = gray;

                Mat img;
                try
                {
                    img = CvInvoke.Imread(openFileDialog.FileName);
                }
                catch
                {
                    MessageBox.Show(String.Format("Invalide File: {0}", openFileDialog.FileName));
                    return;
                }
                Mat down = new Mat();
                CvInvoke.PyrDown(img, down);
                CvInvoke.PyrDown(down, down);
                CvInvoke.PyrUp(down, down);
                UMat uImg = down.GetUMat(AccessType.ReadWrite);
                down.Save(@"C:\Users\MingHan\TestPhoto\Compressed\" + openFileDialog.SafeFileName);
                imageBox1.Image = uImg;
                Thread workerThread = new Thread(()=> ProcessImageAsync(openFileDialog.SafeFileName));
                workerThread.IsBackground = true;
                workerThread.Start();
               
                 //ProcessImageAsync(openFileDialog.SafeFileName);
            }
        }

        //private void ProcessImageAsync(IInputOutputArray image)
        private void ProcessImageAsync(string @image)
        {
            Stopwatch watch = Stopwatch.StartNew(); // time the detection process
            Task.Run(async () =>
            {
                var cognitiveService = new ImageToTextInterpreter
                {
                    ImageFilePath = image,
                    //SubscriptionKey = "31f7ce7f00d94f478b1bfcf192b6cfbd"
                    SubscriptionKey = "59b7a03e08ed4126809dd2d087e4fd81"
                };

                var results = await cognitiveService.ConvertImageToStreamAndExtractText();
                watch.Stop();
                Console.WriteLine(String.Format("License Plate Recognition time: {0} milli-seconds", watch.Elapsed.TotalMilliseconds));
                //OutputToConsole(results, @image, watch);
            }).Wait();

            //Stopwatch watch = Stopwatch.StartNew(); // time the detection process

            //List<IInputOutputArray> licensePlateImagesList = new List<IInputOutputArray>();
            //List<IInputOutputArray> filteredLicensePlateImagesList = new List<IInputOutputArray>();
            //List<RotatedRect> licenseBoxList = new List<RotatedRect>();
            //List<string> words = _licensePlateDetector.DetectLicensePlate(
            //   image,
            //   licensePlateImagesList,
            //   filteredLicensePlateImagesList,
            //   licenseBoxList);

            //watch.Stop(); //stop the timer
            //label1.Text = String.Format("License Plate Recognition time: {0} milli-seconds", watch.Elapsed.TotalMilliseconds);

            //panel1.Controls.Clear();
            //Point startPoint = new Point(10, 10);
            //String licensePlateNumber = "";
            //for (int i = 0; i < words.Count; i++)
            //{
            //    Mat dest = new Mat();
            //    CvInvoke.VConcat(licensePlateImagesList[i], filteredLicensePlateImagesList[i], dest);
            //    AddLabelAndImage(
            //       ref startPoint,
            //       String.Format("License: {0}", words[i]),
            //       dest);
            //    licensePlateNumber += words[i];
            //    PointF[] verticesF = licenseBoxList[i].GetVertices();
            //    Point[] vertices = Array.ConvertAll(verticesF, Point.Round);
            //    using (VectorOfPoint pts = new VectorOfPoint(vertices))
            //        CvInvoke.Polylines(image, pts, true, new Bgr(System.Drawing.Color.Blue).MCvScalar, 2);
            //    //System.Console.WriteLine(words[i]);
            //}

            //await StoreDBAsync(licensePlateNumber);
        }

        //private void AddLabelAndImage(ref Point startPoint, String labelText, IImage image)
        private void AddLabelAndImage(ref Point startPoint, String labelText)
        {
            Label label = new Label();
            panel1.Controls.Add(label);
            label.Text = labelText;
            label.Width = 100;
            label.Height = 30;
            label.Location = startPoint;
            startPoint.Y += label.Height;

            //ImageBox box = new ImageBox();
            //panel1.Controls.Add(box);
            //box.ClientSize = image.Size;
            //box.Image = image;
            //box.Location = startPoint;
            //startPoint.Y += box.Height + 10;
        }

            //private void OutputToConsole(OcrResults results, string imageName, Stopwatch watch)
            private void OutputToConsole(AnnotateImageResponse responses, string imageName, Stopwatch watch)
            {
            //Console.WriteLine("Interpreted text:");
            //Console.ForegroundColor = ConsoleColor.Yellow;
            string ss = "";
            string words = "";
            var keywords = responses?.TextAnnotations?.Select(s => s.Description).ToArray();
            if (keywords != null)
            {
                words = String.Join(" ", keywords);
            }

            //bool match = false;
            //foreach (var region in results.Regions)
            //{
            //    Console.WriteLine(region.Lines.Length);
            //    foreach (var line in region.Lines)
            //    {
            //        s = s + " " + string.Join(" ", line.Words.Select(w => w.Text));
            //        Console.WriteLine(string.Join(" ", line.Words.Select(w => w.Text)));
            //    }
            //}
            MatchCollection mc = Regex.Matches(words.ToString(), @"([JKKDMNCPARBTSQVWL]){1}\w{0,}\s{0,1}\d{1,4}([A-Za-z])?");

            bool match = false;
            foreach (Match m in mc)
            {
                match = true;
                ss = m.ToString();
            }
            Console.WriteLine("Regex: " + ss);
            //Console.ForegroundColor = ConsoleColor.White;
            //Console.ReadLine();
            if (match)
            {
                this.label2.Invoke((MethodInvoker)delegate
                {
                    this.label2.Text = ss;
                    this.label3.Text = String.Format("License Plate Recognition time: {0} milli-seconds", watch.Elapsed.TotalMilliseconds);
                });
                StoreDBAsync(imageName, ss);
            }
            var readyTask = Task.Run(async() =>
            {
                await Task.Delay(3800);
                ready = true;
                //_capture.Start();
            });
            //if (IsHandleCreated)
            //    this.CreateControl();
            //Control control = new Control();
            //control.Invoke(new Action(() => label2.Text = s));
            //label2.Text = s;
            //Point startPoint = new Point(10, 10);
            //AddLabelAndImage(ref startPoint,String.Format("License: {0}", s));
        }

        private async void StoreDBAsync(string @image, string plate)
        {
            //var stream = File.Open(@"C:\Users\MingHan\TestPhoto\Compressed\" + image, FileMode.Open);
            var stream = File.Open(image, FileMode.Open);
            var task = new FirebaseStorage("park-e5cd7.appspot.com")
                .Child("car")
                .Child(image)
                .PutAsync(stream);

            //task.Progress.ProgressChanged += (s, e) => Console.WriteLine($"Progress: {e.Percentage} %");
            var downloadUrl = await task;

            plate = RemoveSpecialCharacters(plate);

            Car newCar = new Car();
            newCar.CarNumber = plate;
            newCar.LastEnterTime = DateTime.Now.ToString("HH:mm:ss");
            newCar.LastEnterDate = DateTime.Now.ToString("dd/MM/yyyy");
            newCar.timestamp = new Dictionary<string, object> { { ".sv", "timestamp" } };

            Stat stat = new Stat();
            stat.carNumber = plate;
            stat.timestamp = new Dictionary<string, object> { { ".sv", "timestamp" } };

            var firebase = new FirebaseClient("https://park-e5cd7.firebaseio.com/");
            await firebase
                .Child("car")
                .Child(newCar.CarNumber)
                .PutAsync(newCar);

            var carRecord = await firebase
                .Child("record")
                .Child(newCar.CarNumber)
                .Child("record")
                .PostAsync(newCar);

            await firebase
                .Child("statCar")
                .PostAsync(stat);
        }

        private string RemoveSpecialCharacters(string str)
        {
            return Regex.Replace(str, "[^a-zA-Z0-9]+", "", RegexOptions.Compiled);
        }

        public class ImageToTextInterpreter
        {
            public string ImageFilePath { get; set; }

            public string SubscriptionKey { get; set; }

            const string UNKNOWN_LANGUAGE = "unk";

            public async Task<OcrResults> ConvertImageToStreamAndExtractText()
            {
                var visionServiceClient = new VisionServiceClient(SubscriptionKey);

                using (Stream imageFileStream = File.OpenRead(ImageFilePath))
                {
                    return await visionServiceClient.RecognizeTextAsync(imageFileStream, UNKNOWN_LANGUAGE);
                }
            }
        }

        protected override void Dispose(bool disposing)
        {
             if (disposing && (components != null))
             {
                components.Dispose();
             }
             base.Dispose(disposing);
        }
    }
}
