
// Initialize the map
var map = L.map('map').setView([-37.8136, 144.9631], 11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// function findFastestPath(startPoint, endPoint) {
//   // Mock data: Simplified path with lat/lng coordinates
//   var paths = {
//     '970': { lat: -37.86703, lng: 145.09159 },
//     '2000': { lat: -37.8516827, lng: 145.0943457 },
//     '2200': { lat: -37.81631, lng: 145.09812 },
//   };  

//   var startPointCoords = paths[startPoint];
//   var endPointCoords = paths[endPoint];

//   if (!startPointCoords || !endPointCoords) {
//       alert('Invalid start or end point selection.');
//       return [];
//   }
//   return [startPointCoords, endPointCoords];
// }

var currentRoute;

document.getElementById("form").addEventListener("submit", async function () {
  event.preventDefault();
  // const data = new FormData(this);

  // var startPoint = document.getElementById("startScats").value;
  // var endPoint = document.getElementById("endScats").value;
  // var selectedModel = document.getElementById("models").value;
  // var selectedTime = document.getElementById("time").value;
  // var selectedDate = document.getElementById("date").value;

  const params = {
    'startScats': startScats,
    'endScats': endScats,
    'models': models,
    'time': time,
    'date': date
  }
  
  const url = 'http://localhost:3001';
  
  const routeInfo = await axios.post(url, _, {params})
    .then(response => {
      console.log('Response:', response.data);
    })
    .catch(error => {
      console.error('Error:', error);
  });
  
  if (currentRoute) {
    map.removeLayer(currentRoute);
  }

  // TODO: change this to be the model shit 
  // var fastestPathCoordinates = findFastestPath(startPoint, endPoint);

  // TODO: make 4 - 1 blue, 3 grey
  var path = L.polyline(routeInfo, { color: 'blue' }).addTo(map);
  // Fit the map to the bounds of the path
  map.fitBounds(path.getBounds());

  currentRoute = path;
});


