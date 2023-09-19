$(() => {
  // initialize UI
  $("#warning").hide();
  $("#compare-button").attr("disabled", "disabled");

  // show selected image
  $("#inputImage1").change(function (e) {
    const file = $("#inputImage1")[0].files[0];
    const img = createImage(file);
    const cardBody = createCardBody(file.name);
    $("#dummyImage1").remove();
    $("#image1Div").empty();
    $("#image1Div").append(img, cardBody);
    checkInputImage();
  });

  $("#inputImage2").change(function (e) {
    const file = $("#inputImage2")[0].files[0];
    const img = createImage(file);
    const cardBody = createCardBody(file.name);
    $("#dummyImage2").remove();
    $("#image2Div").empty();
    $("#image2Div").append(img, cardBody);
    checkInputImage();
  });

  // compare images
  $("#compare-button").click(async () => {
    // // $("#warning").show();
    $("#compare-button").attr("disabled", "disabled");

    const response = await postCompareImage();

    const title1 = "Objects appear in picture 2 but do not appear in picture 1";
    const title2 = "Objects appear in picture 1 but do not appear in picture 2";

    const img1 = createResultImage(response["result1"], title1);
    const img2 = createResultImage(response["result2"], title2);

    $("#resultImage1").empty();
    $("#resultImage2").empty();

    $("#resultImage1").append(title1, img1);
    $("#resultImage2").append(title2, img2);

    $("#compare-button").removeAttr("disabled");
  });
});

const checkInputImage = () => {
  if ($("#inputImage1")[0].files[0] && $("#inputImage2")[0].files[0]) {
    $("#compare-button").removeAttr("disabled");
  } else {
    $("#compare-button").attr("disabled", "disabled");
  }
};

const createImage = (file) => {
  var img = document.createElement("img");
  var reader = new FileReader();
  reader.onloadend = function () {
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
  img.classList.add("card-img-top");
  img.setAttribute("title", file.name);
  return img;
};

const createCardBody = (filename) => {
  // return "<div><p>" + filename + "</p></div>";
  return '<div class="">File name: ' + filename + "</div>";
};

// call compare api
const postCompareImage = async () => {
  let formData = new FormData();
  formData.append("image1", $("#inputImage1")[0].files[0]);
  formData.append("image2", $("#inputImage2")[0].files[0]);

  // get menue recommendation from API
  try {
    const { data } = await axios.post("/compare-image", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return data;
  } catch (error) {
    console.error(`failed to get response from gpt: ${error}`);
  }
};

const createResultImage = (filename, title = "") => {
  return (
    '<img class="card-img-top" title="' +
    title +
    '" src="/image-data/' +
    filename +
    '" />'
  );
};

const dummyDiv = (divId) => {
  return (
    '<div class="border border-secondary" style="min-height: 100px" id="' +
    divId +
    '"></div>'
  );
};
