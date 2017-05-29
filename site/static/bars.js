
var svgContainer = d3.select("#canvas").append("svg")
                                    .attr("width", 1000)
                                    .attr("height", 400);

var url = "http://thezoo.noip.me:5000/query"
d3.json(url, function (json) {
  var rects = svgContainer.selectAll("rect")
	                           .data(json.probs)
	                           .enter()
	                           .append("rect")
  var rectAttributes = rects 
                       .attr("x", function (d, i) { return i; })
                       .attr("y", 10)
                       .attr("height", function (d) { return Math.round(1000*d); })
		       .attr("width", 1)
                       .style("fill", "red");
})
