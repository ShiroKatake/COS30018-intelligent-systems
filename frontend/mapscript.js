// Initialize the map
let map = L.map("map").setView([-37.8136, 144.9631], 11);

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

  const startPoint = document.getElementById("start_scat").value;
  const endPoint = document.getElementById("end_scat").value;
  const selectedModel = document.getElementById("model").value;
  const selectedTime = document.getElementById("time").value;
  const selectedDate = document.getElementById("date").value;

  const routeInfo = await getRoutes(params);

  // Initialize polyline array
  let polylines = [];

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
      const [_, value] = Object.entries(location)[0];
      const lat = value.lat;
      const lng = value.long;
      pathCoordinates.push([lat, lng]);
    });

    const pathColour = index === 0 ? "blue" : "grey";
    let path = L.polyline(pathCoordinates, { color: pathColour });

    path.addTo(map);
    polylines.push(path);
  });

  if (polylines.length > 0) {
    map.fitBounds(L.featureGroup(polylines).getBounds());
  }
});
