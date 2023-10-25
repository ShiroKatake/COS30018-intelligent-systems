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

const form = document.getElementById("form");
form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const startPoint = document.getElementById("start_scat").value;
  const endPoint = document.getElementById("end_scat").value;
  const selectedModel = document.getElementById("model").value;
  const selectedTime = document.getElementById("time").value;
  const selectedDate = document.getElementById("date").value;

  const params = {
    start_scat: startPoint,
    end_scat: endPoint,
    model: selectedModel,
    time: selectedTime,
    date: selectedDate,
  };

  const routeInfo = await getRoutes(params);

  // Initialize polyline array
  var polylines = [];
  // Selected path
  var selectedPath = null; 
  // Map markers 
  var originMarker;
  var destinationMarker;

  // Delete old paths and markers
  polylines.forEach((polyline) => {
    map.removeLayer(polyline);
  });

  // Delete old map markers
  if (originMarker) {
    map.removeLayer(originMarker);
  }
  if (destinationMarker) {
    map.removeLayer(destinationMarker);
  }


  // Clear the arrays
  polylines = [];
  originMarker = null;
  destinationMarker = null;

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

    // Put an origin/dest marker on the first path
    if (index === 0) {
      const firstLocation = route.route[0];
      const lastLocation = route.route[route.route.length - 1];

      // Origin marker
      originMarker = L.marker([
        firstLocation[Object.keys(firstLocation)[0]].lat,
        firstLocation[Object.keys(firstLocation)[0]].long,
      ])
        .bindPopup("Origin")
        .addTo(map);

      // Destination marker
      destinationMarker = L.marker([
        lastLocation[Object.keys(lastLocation)[0]].lat,
        lastLocation[Object.keys(lastLocation)[0]].long,
      ])
        .bindPopup("Destination")
        .addTo(map);
    }

    // Add click event to swap between paths
    path.on("click", function () {
      if (selectedPath) {
        // Old selected path 
        selectedPath.setStyle({ color: "grey" });
      }

      // New chosen path
      path.setStyle({ color: "blue" });
      selectedPath = path;
    });

    // First path selected by default
    if (index === 0) {
      selectedPath = path;
    }
  });

  if (polylines.length > 0) {
    map.fitBounds(L.featureGroup(polylines).getBounds());
  }
});
