// Initialize the map
var map = L.map("map").setView([-37.8136, 144.9631], 11);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const getRoutes = async (params) => {
  try {
    const url = "http://localhost:3001";
    const response = await axios.post(url, null, { params });
    const { data } = response;
    return data;
  } catch (error) {
    console.log(error);
  }
}

  event.preventDefault();

  // var startPoint = document.getElementById("start_scat").value;
  // var endPoint = document.getElementById("end_scat").value;
  // var selectedModel = document.getElementById("model").value;
  // var selectedTime = document.getElementById("time").value;
  // var selectedDate = document.getElementById("date").value;

  // const params = {
  //   start_scat: startPoint,
  //   end_scat: endPoint,
  //   model: selectedModel,
  //   time: selectedTime,
  //   date: selectedDate,
  // };

  // const url = "http://localhost:3001";

  // const { data } = await axios
  //   .post(url, null, { params })
  //   .then((response) => {
  //     console.log("Response:", response.data);
  //   })
  //   .catch((error) => {
  //     console.error("Error:", error);
  //   });

  // if (currentRoute) {
  //   map.removeLayer(currentRoute);
  // }

  // const routeInfo = data;

  //TODO: return lat long of the first route return red
  //TODO: return the others as grey
  const routeInfo = [
    {
      route: [
        {
          970: {
            name: "WARRIGAL RD/HIGH STREET RD",
            lat: -37.86703,
            long: 145.09159,
          },
        },
        {
          3685: {
            name: "WARRIGAL RD/HIGHBURY RD",
            lat: -37.85467,
            long: 145.09384,
          },
        },
        {
          2000: {
            name: "WARRIGAL RD/TOORAK RD",
            lat: -37.8516827,
            long: 145.0943457,
          },
        },
        {
          4043: {
            name: "BURKE RD/TOORAK RD",
            lat: -37.84683,
            long: 145.05275,
          },
        },
        {
          4040: {
            name: "BURKE RD/RIVERSDALE RD",
            lat: -37.83256,
            long: 145.05545,
          },
        },
        {
          4266: {
            name: "BURWOOD RD/AUBURN RD",
            lat: -37.82529,
            long: 145.04387,
          },
        },
        {
          4264: {
            name: "GLENFERRIE RD/BURWOOD RD",
            lat: -37.82389,
            long: 145.03409,
          },
        },
        {
          4324: {
            name: "COTHAM RD/GLENFERRIE RD",
            lat: -37.809274,
            long: 145.037306,
          },
        },
        {
          3662: {
            name: "PRINCESS ST/HIGH ST",
            lat: -37.80876,
            long: 145.02757,
          },
        },
        {
          2820: {
            name: "EARL ST/PRINCESS ST",
            lat: -37.79477,
            long: 145.03077,
          },
        },
      ],
      travel_time: 17.781896047690434,
    },
    {
      route: [
        {
          970: {
            name: "WARRIGAL RD/HIGH STREET RD",
            lat: -37.86703,
            long: 145.09159,
          },
        },
        {
          3685: {
            name: "WARRIGAL RD/HIGHBURY RD",
            lat: -37.85467,
            long: 145.09384,
          },
        },
        {
          2000: {
            name: "WARRIGAL RD/TOORAK RD",
            lat: -37.8516827,
            long: 145.0943457,
          },
        },
        {
          4043: {
            name: "BURKE RD/TOORAK RD",
            lat: -37.84683,
            long: 145.05275,
          },
        },
        {
          4040: {
            name: "BURKE RD/RIVERSDALE RD",
            lat: -37.83256,
            long: 145.05545,
          },
        },
        {
          4266: {
            name: "BURWOOD RD/AUBURN RD",
            lat: -37.82529,
            long: 145.04387,
          },
        },
        {
          4264: {
            name: "GLENFERRIE RD/BURWOOD RD",
            lat: -37.82389,
            long: 145.03409,
          },
        },
        {
          4324: {
            name: "COTHAM RD/GLENFERRIE RD",
            lat: -37.809274,
            long: 145.037306,
          },
        },
        {
          3662: {
            name: "PRINCESS ST/HIGH ST",
            lat: -37.80876,
            long: 145.02757,
          },
        },
        {
          4335: {
            name: "HIGH ST/CHARLES ST",
            lat: -37.80624,
            long: 145.03518,
          },
        },
        {
          4321: {
            name: "HIGH ST/HARP ST",
            lat: -37.800776,
            long: 145.0494611,
          },
        },
        {
          2820: {
            name: "EARL ST/PRINCESS ST",
            lat: -37.79477,
            long: 145.03077,
          },
        },
      ],
      travel_time: 21.102222433368446,
    },
    {
      route: [
        {
          970: {
            name: "WARRIGAL RD/HIGH STREET RD",
            lat: -37.86703,
            long: 145.09159,
          },
        },
        {
          3685: {
            name: "WARRIGAL RD/HIGHBURY RD",
            lat: -37.85467,
            long: 145.09384,
          },
        },
        {
          2000: {
            name: "WARRIGAL RD/TOORAK RD",
            lat: -37.8516827,
            long: 145.0943457,
          },
        },
        {
          4043: {
            name: "BURKE RD/TOORAK RD",
            lat: -37.84683,
            long: 145.05275,
          },
        },
        {
          4040: {
            name: "BURKE RD/RIVERSDALE RD",
            lat: -37.83256,
            long: 145.05545,
          },
        },
        {
          4266: {
            name: "BURWOOD RD/AUBURN RD",
            lat: -37.82529,
            long: 145.04387,
          },
        },
        {
          4264: {
            name: "GLENFERRIE RD/BURWOOD RD",
            lat: -37.82389,
            long: 145.03409,
          },
        },
        {
          4324: {
            name: "COTHAM RD/GLENFERRIE RD",
            lat: -37.809274,
            long: 145.037306,
          },
        },
        {
          3662: {
            name: "PRINCESS ST/HIGH ST",
            lat: -37.80876,
            long: 145.02757,
          },
        },
        {
          4335: {
            name: "HIGH ST/CHARLES ST",
            lat: -37.80624,
            long: 145.03518,
          },
        },
        {
          4321: {
            name: "HIGH ST/HARP ST",
            lat: -37.800776,
            long: 145.0494611,
          },
        },
        {
          4030: {
            name: "BURKE RD/DONCASTER RD",
            lat: -37.79561,
            long: 145.06251,
          },
        },
        {
          2825: {
            name: "BURKE RD/EASTERN_FWY",
            lat: -37.78661,
            long: 145.06202,
          },
        },
        {
          2820: {
            name: "EARL ST/PRINCESS ST",
            lat: -37.79477,
            long: 145.03077,
          },
        },
      ],
      travel_time: 25.508124065439738,
    },
    {
      route: [
        {
          970: {
            name: "WARRIGAL RD/HIGH STREET RD",
            lat: -37.86703,
            long: 145.09159,
          },
        },
        {
          3685: {
            name: "WARRIGAL RD/HIGHBURY RD",
            lat: -37.85467,
            long: 145.09384,
          },
        },
        {
          2000: {
            name: "WARRIGAL RD/TOORAK RD",
            lat: -37.8516827,
            long: 145.0943457,
          },
        },
        {
          4043: {
            name: "BURKE RD/TOORAK RD",
            lat: -37.84683,
            long: 145.05275,
          },
        },
        {
          4040: {
            name: "BURKE RD/RIVERSDALE RD",
            lat: -37.83256,
            long: 145.05545,
          },
        },
        {
          4266: {
            name: "BURWOOD RD/AUBURN RD",
            lat: -37.82529,
            long: 145.04387,
          },
        },
        {
          4264: {
            name: "GLENFERRIE RD/BURWOOD RD",
            lat: -37.82389,
            long: 145.03409,
          },
        },
        {
          4324: {
            name: "COTHAM RD/GLENFERRIE RD",
            lat: -37.809274,
            long: 145.037306,
          },
        },
        {
          3662: {
            name: "PRINCESS ST/HIGH ST",
            lat: -37.80876,
            long: 145.02757,
          },
        },
        {
          4335: {
            name: "HIGH ST/CHARLES ST",
            lat: -37.80624,
            long: 145.03518,
          },
        },
        {
          4321: {
            name: "HIGH ST/HARP ST",
            lat: -37.800776,
            long: 145.0494611,
          },
        },
        {
          4032: {
            name: "BURKE RD/HARP RD",
            lat: -37.80202,
            long: 145.06127,
          },
        },
        {
          4030: {
            name: "BURKE RD/DONCASTER RD",
            lat: -37.79561,
            long: 145.06251,
          },
        },
        {
          2825: {
            name: "BURKE RD/EASTERN_FWY",
            lat: -37.78661,
            long: 145.06202,
          },
        },
        {
          2820: {
            name: "EARL ST/PRINCESS ST",
            lat: -37.79477,
            long: 145.03077,
          },
        },
      ],
      travel_time: 26.4926930061878,
    },
  ];

  // var path = L.polyline(routeInfo, { color: "blue" }).addTo(map);

  // Create a polyline using the coordinates
  // var path = L.polyline(coordinates, { color: "blue" });

  // Add the polyline to the map
  // path.addTo(map);

  // routeInfo[0], color: red
  // routeInfo[1], color: grey

  // Initialize polyline array
  var polylines = [];

  // Delete old paths
  polylines.forEach((polyline) => {
    map.removeLayer(polyline);
  });

  // Clear the array
  polylines = [];

  // Iterate over the routeInfo to return lat and longs of each route
  routeInfo.forEach((route, index) => {
    const pathCoordinates = [];

    route.route.forEach((location) => {
      const [key, value] = Object.entries(location)[0];
      const lat = value.lat;
      const lng = value.long;
      pathCoordinates.push([lat, lng]);
    });

    const pathColour = index === 0 ? "blue" : "grey";
    var path = L.polyline(pathCoordinates, { color: pathColour });

    path.addTo(map);
    polylines.push(path);
  });

  if (polylines.length > 0) {
    map.fitBounds(L.featureGroup(polylines).getBounds());
  }
});
