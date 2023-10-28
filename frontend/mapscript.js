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

// Initialize 
let polylines = [];
let originMarker;
let destinationMarker;

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

  console.log("Calculating routes...");
  const routeResult = await getRoutes(params);

  // Delete old paths
  polylines.forEach((polyline) => {
    polyline.remove();
  });

  // Delete old map markers
  if (originMarker) {
    originMarker.remove();
  }
  if (destinationMarker) {
    destinationMarker.remove();
  }

  // Clear the arrays
  polylines = [];
  originMarker = null;
  destinationMarker = null;

  // Iterate over the routeInfo to return lat and longs of each route
  routeResult.forEach((route, index) => {
    logDirections(route.directions, index);

    // Add path, best path set to blue
    const pathColour = index === 0 ? "blue" : "grey";
    let path = L.polyline(route.lat_long_route, { color: pathColour });

    path.addTo(map);
    polylines.push(path);
    path.bringToBack();

    // Add travel time estimation to the path
    path.bindPopup(`${Math.round(route.travel_time)} minutes`);

    // Put an origin/dest marker on the first path
    if (index === 0) {
      const markers = placeMarkers(route);
      originMarker = markers[0];
      destinationMarker = markers[1];

      path.openPopup();
    }

    // Add click event to swap between paths
    path.on("click", function () {
      polylines.forEach((polyline) => {
        if (polyline._leaflet_id !== path._leaflet_id) {
          polyline.setStyle({ color: "grey" });
        }
      });

      // New chosen path
      path.setStyle({ color: "blue" });
      path.bringToFront();
    });
  });

  if (polylines.length > 0) {
    map.fitBounds(L.featureGroup(polylines).getBounds());
  }
});

const logDirections = (directions, index) => {
  // Outputs route number
  let route = `===== Route ${index + 1} =====`;

  // Outputs intersection's scat number and name in order
  directions.forEach((direction) => {
    route += `\n${Object.keys(direction)}: ${direction[Object.keys(direction)]}`;
  });
  console.log(route);
}

// TODO: Go to group therapy
const placeMarkers = (route) => {
  const {
    directions,     // We take directions array because it has scat numbers and names
    lat_long_route  // We take lat_long_route array because it has lat and long of each intersection
  } = route;

  // Origin marker
  originMarker = L.marker([
    lat_long_route[0][0], // First index of the [lat, long] pair is lat
    lat_long_route[0][1], // Second index of the [lat, long] pair is long
  ])
    .bindPopup(`Origin SCAT ${Object.keys(directions[0])}`) // First index of 'directions' is the origin
    .addTo(map);

  // Destination marker
  destinationMarker = L.marker([
    lat_long_route[lat_long_route.length - 1][0], // First index of the [lat, long] pair is lat
    lat_long_route[lat_long_route.length - 1][1], // Second index of the [lat, long] pair is long
  ])
    .bindPopup(`Destination SCAT ${Object.keys(directions[directions.length - 1])}`)  // Last index of 'directions' is the destination
    .addTo(map);
  
  // Return markers to keep track
  return [originMarker, destinationMarker];
}