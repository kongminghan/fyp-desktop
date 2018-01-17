using Emgu.CV;
using System;
using System.Collections.Generic;
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
using System.Drawing;
using Google.Apis.Auth.OAuth2;
using Google.Apis.Vision.v1;
using Google.Apis.Http;
using Google.Apis.Services;
using Google.Apis.Vision.v1.Data;
using System.Net;
using Newtonsoft.Json;
using FireSharp.Config;
using FireSharp.Interfaces;
using FireSharp.Response;
using System.Timers;

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
        GoogleCredential credentails;
        VisionService service;
        private string mall = "" ;
        private int count = 0;
        private string key;
        private List<string> passList = new List<string>();

        OpenFileDialog openFileDialog = new OpenFileDialog();

        public Form1()
        {
            InitializeComponent();
            CvInvoke.UseOpenCL = false;

            //create service
            credentails = CreateCredentials("service_account.json");
            service = CreateService("Park", credentails);

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
            }

            PutInitialAsync();
            System.Timers.Timer aTimer = new System.Timers.Timer();
            aTimer.Elapsed += new ElapsedEventHandler(OnTimedEventAsync);
            aTimer.Interval = 900000;
            aTimer.Enabled = true;
        }

        private async void PutInitialAsync()
        {
            var firebase = new Firebase.Database.FirebaseClient("https://park-e5cd7.firebaseio.com/");

            var mallList = await firebase
              .Child("user")
              .OnceAsync<User>();

            foreach (var list in mallList)
            {
                comboBox1.Items.Add(list.Key);
                passList.Add(list.Object.password);
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
                //File.Delete(filename);
                Task.Run(() =>
                {
                    File.Delete(filename);
                });
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
        }

            //private void OutputToConsole(OcrResults results, string imageName, Stopwatch watch)
            private void OutputToConsole(AnnotateImageResponse responses, string imageName, Stopwatch watch)
            {
            string ss = "";
            string words = "";
            var keywords = responses?.TextAnnotations?.Select(s => s.Description).ToArray();
            if (keywords != null)
            {
                words = String.Join(" ", keywords);
            }

            MatchCollection mc = Regex.Matches(words.ToString(), @"([1]){0,1}([JKKDMNCPARBTSQVWL]){1}\w{0,}\s{0,1}\d{1,4}([A-Za-z])?");

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
                await Task.Delay(3500);
                ready = true;
                //_capture.Start();
            });
        }

        private async void StoreDBAsync(string @image, string plate)
        {
            ++count;
            plate = RemoveSpecialCharacters(plate);

            Car newCar = new Car();
            newCar.CarNumber = plate;
            newCar.LastEnterTime = DateTime.Now.ToString("HH:mm:ss");
            newCar.LastEnterDate = DateTime.Now.ToString("dd/MM/yyyy");
            newCar.timestamp = new Dictionary<string, object> { { ".sv", "timestamp" } };
            newCar.CarLocation = mall;

            var firebase = new Firebase.Database.FirebaseClient("https://park-e5cd7.firebaseio.com/");

            IFirebaseConfig config = new FirebaseConfig
            {
                AuthSecret = AUTH_SECRET,
                BasePath = "https://park-e5cd7.firebaseio.com/"
            };

            IFirebaseClient client = new FireSharp.FirebaseClient(config);
            var token = new rate();

            FirebaseResponse response = await client.GetAsync("car/"+plate);
            token = response.ResultAs<rate>(); //The response will contain the data being retreived

            if (token != null)
            {
                newCar.CMToken = token.CMToken;
                SendPushNotification(token.CMToken);
            }

            await firebase
                .Child("car")
                .Child(newCar.CarNumber)
                .PutAsync(newCar);

            var carRecord = await firebase
                .Child("record")
                .Child(newCar.CarNumber)
                .Child("record")
                .PostAsync(newCar);

            var statCar = await firebase
                .Child("mall")
                .Child(mall)
                .PostAsync(newCar);

            Stat stat = new Stat();
            stat.count = count;
            stat.timestamp = new Dictionary<string, object> { { ".sv", "timestamp" } };

            await firebase
                .Child("usage")
                .Child(mall)
                .Child(key)
                .PutAsync(stat);
        }

        private string RemoveSpecialCharacters(string str)
        {
            return Regex.Replace(str, "[^a-zA-Z0-9]+", "", RegexOptions.Compiled);
        }

        public Int64 GetTimestamp(DateTime value)
        {
            return Convert.ToInt64(value.ToString("yyyyMMddHHmmssffff"));
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

        public void SendPushNotification(string deviceId)
        {
            try
            {

                string applicationID = "AAAAwkPRZh0:APA91bEEtEi0CnQ9qoDlFm6n4KgTt7NXbWMLJ3XLWAt_ZR2kvvKogQfNecE2wT9-bY2FfIlKVap74yn4jq0BwWYNkgfpQd8N1AUTo2PHjBWxtuL2VRoDXM8RaNA3zWobRT-x7PIMklv4";

                string senderId = "834361452061";

                WebRequest tRequest= WebRequest.Create("https://fcm.googleapis.com/fcm/send");

                tRequest.Method = "post";
                tRequest.ContentType = "application/json";
                var data = new
                {
                    to = deviceId,
                    notification = new
                    {
                        body = "You car has just entered " + mall + " on " + DateTime.Now.ToString("HH:mm:ss") +" today.",
                        title = "Notification from VPS",
                        sound = "Enabled",
                        icon = "ic_local_parking_black_24dp"
                    }
                };
                
                var json = JsonConvert.SerializeObject(data);
                Byte[] byteArray = Encoding.UTF8.GetBytes(json);
                tRequest.Headers.Add(string.Format("Authorization: key={0}", applicationID));
                tRequest.Headers.Add(string.Format("Sender: id={0}", senderId));
                tRequest.ContentLength = byteArray.Length;
                using (Stream dataStream = tRequest.GetRequestStream())
                {
                    dataStream.Write(byteArray, 0, byteArray.Length);
                    using (WebResponse tResponse = tRequest.GetResponse())
                    {
                        using (Stream dataStreamResponse = tResponse.GetResponseStream())
                        {
                            using (StreamReader tReader = new StreamReader(dataStreamResponse))
                            {
                                String sResponseFromServer = tReader.ReadToEnd();
                                string str = sResponseFromServer;
                                Console.WriteLine(str);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                string str = ex.Message;
                Console.WriteLine(str);
            }
        }

        private async void button2_ClickAsync(object sender, EventArgs e)
        {
            if (comboBox1.GetItemText(comboBox1.SelectedItem) != "")
            {
                //MessageBox.Show(comboBox1.GetItemText(comboBox1.SelectedItem));
                mall = comboBox1.GetItemText(comboBox1.SelectedItem);
                var firebase = new Firebase.Database.FirebaseClient("https://park-e5cd7.firebaseio.com/");

                if (textBox1.Text.Length > 0)
                {
                    if (textBox1.Text == passList[comboBox1.SelectedIndex])
                    {
                        _capture.Start();
                        comboBox1.Enabled = false;

                        Stat stat = new Stat();
                        stat.count = 0;
                        stat.timestamp = new Dictionary<string, object> { { ".sv", "timestamp" } };

                        var usage = await firebase
                            .Child("usage")
                            .Child(mall)
                            .PostAsync(stat);

                        key = usage.Key;
                    }
                    else
                    {
                        MessageBox.Show("Wrong password! Please contact your manager.");
                    }
                }
                else
                {
                    MessageBox.Show("Please enter password!");
                }
            }
            else
            {
                MessageBox.Show("Please select a mall from the combobox!");
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            comboBox1.Enabled = true;
            _capture.Stop();
            //_capture.Dispose();
        }

        private async void OnTimedEventAsync(object source, ElapsedEventArgs e)
        {
            Stat stat = new Stat();
            stat.count = 0;
            //stat.carNumber = plate;
            //stat.LastEnterTime = newCar.LastEnterTime;
            //stat.LastEnterDate = newCar.LastEnterDate;
            stat.timestamp = new Dictionary<string, object> { { ".sv", "timestamp" } };
            var firebase = new Firebase.Database.FirebaseClient("https://park-e5cd7.firebaseio.com/");
            var usage = await firebase
                .Child("usage")
                .Child(mall)
                .PostAsync(stat);

            key = usage.Key;
            count = 0;
        }
    }
}
