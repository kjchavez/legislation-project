
var svgContainer = d3.select("#canvas").append("svg")
                                    .attr("width", 1000)
                                    .attr("height", 400);
var textContainer = d3.select("#sample-text");

// Controllers for generated text
var tempSlider = d3.select("#temp")
var seedTextBox = d3.select("#seed")

function sliderToTemp(sliderVal) {
  // Squash the slider to a reasonable range.
  var exp = (sliderVal / 50) - 1;
  return Math.pow(10, exp);
}

function updateText (temp) {
  var generateUrl = "/generate?temp="+temp
  d3.json(generateUrl, function (json) {
    str = json.text.replace(/(?:\r\n|\r|\n)/g, '<br />');
    textContainer.html(str);
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
