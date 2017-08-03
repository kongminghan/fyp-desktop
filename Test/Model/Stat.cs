using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test.Model
{
    class Stat
    {
        public String carNumber { get; set; }
        public string LastEnterDate { get; set; }
        public string LastEnterTime { get; set; }
        public int count { get; set; }
        public Dictionary<string, object> timestamp { get; set; }
    }
}
