using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test.Model
{
    class Car
    {
        public string CarNumber { get; set; }
        public string LastEnterDate { get; set; }
        public string LastEnterTime { get; set; }
        public string Status { get; set; }
        public Dictionary<string, object> timestamp { get; set;  }
        //public string LastExitDate { get; set; }
        //public string LastExitTime { get; set; }

        //public bool Pay { get; set; }
    }
}
