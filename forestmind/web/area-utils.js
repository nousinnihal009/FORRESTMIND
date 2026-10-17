/* Pure coordinate helpers shared by the map and its keyboard-accessible inputs. */
(function (root) {
  'use strict';
  const R = 6371.0088;
  function areaKm2(bbox) {
    const [w,s,e,n] = bbox, radians = Math.PI / 180;
    return R * R * (e-w) * radians * (Math.sin(n*radians)-Math.sin(s*radians));
  }
  function validateBbox(bbox, maxArea = 25) {
    if (!Array.isArray(bbox) || bbox.length !== 4 || bbox.some(v => typeof v !== 'number' || !Number.isFinite(v))) return 'Enter all four finite WGS84 coordinates.';
    const [w,s,e,n] = bbox;
    if (w < -180 || e > 180 || s < -90 || n > 90) return 'Longitude must be −180 to 180 and latitude −90 to 90.';
    if (w >= e) return 'West must be less than east. Areas crossing the antimeridian must be split into separate areas.';
    if (s >= n) return 'South must be less than north.';
    if (Math.abs(s) > 85 || Math.abs(n) > 85) return 'Polar acquisition is unsupported. Choose an area between 85°S and 85°N.';
    if (e-w > 1 || n-s > 1) return 'Select a compact rectangle with each side spanning at most 1°.';
    const area = areaKm2(bbox);
    if (area > maxArea) return `This area is ${area.toFixed(2)} km². Select a smaller rectangle (maximum ${maxArea} km²).`;
    if (area < .0001) return 'Choose an area of at least 0.0001 km².';
    return null;
  }
  function project(lon, lat, zoom) {
    const size = 256 * 2 ** zoom, y = Math.max(-85.05112878,Math.min(85.05112878,lat)) * Math.PI / 180;
    return [(lon+180)/360*size,(1-Math.log(Math.tan(y)+1/Math.cos(y))/Math.PI)/2*size];
  }
  function unproject(x,y,zoom) {
    const size = 256 * 2 ** zoom;
    return [x/size*360-180,Math.atan(Math.sinh(Math.PI*(1-2*y/size)))*180/Math.PI];
  }
  const api = {areaKm2,validateBbox,project,unproject};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ForestArea = api;
})(typeof window === 'undefined' ? globalThis : window);
