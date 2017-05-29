
var svgContainer = d3.select("#canvas").append("svg")
                                    .attr("width", 1000)
                                    .attr("height", 400);
var textContainer = d3.select("#canvas").append("p");

var url = "http://thezoo.noip.me:5000/query"
d3.json(url, function (json) {
  var rects = svgContainer.selectAll("rect")
	                           .data(json.probs)
	                           .enter()
	                           .append("rect");
  var rectAttributes = rects 
                       .attr("x", function (d, i) { return i; })
                       .attr("y", 10)
                       .attr("height", function (d) { return Math.round(1000*d); })
		       .attr("width", 1)
                       .style("fill", "red");
})

// Controllers for generated text
var tempSlider = d3.select("#temp")
var seedTextBox = d3.select("#seed")

function sliderToTemp(sliderVal) {
  // Squash the slider to a reasonable range.
  var exp = (sliderVal / 50) - 1;
  return Math.pow(10, exp);
}

function updateText (temp) {
  var generateUrl = "http://thezoo.noip.me:5000/generate?temp="+temp
  d3.json(generateUrl, function (json) {
    textContainer.text(json.text);
  })
}
// To avoid flooding the server.
updateText = _.debounce(updateText, 300);

function updateTemp (sliderVal) {
  var temp = sliderToTemp(parseFloat(sliderVal));
  tempSlider.property("value", sliderVal)
  d3.select("#temp-value").text(""+temp)
  updateText(temp)
}

// When the temperature changes, re-generate the text.
d3.select("#temp").on("input", function() {
  updateTemp(this.value)
});

updateTemp(50);
