$(() => {
  // initialize UI
  $("#warning").hide();

  // set menu button event
  $("#correct-button").click(async () => {
    // $("#warning").show();
    $("#outputTextarea").val("");
    $("#correct-button").attr("disabled", "disabled");
    // $("#loading_text").html("Processing ...");
    const response = await getResultFromGPT();
    const result = response["content"] ?? "";
    $("#correct-button").disabled = false;
    // $("#warning").hide();
    $("#correct-button").removeAttr("disabled");
    $("#outputTextarea").val(result);
  });
});

// get corrected text from GPT
const getResultFromGPT = async () => {
  const selectedLanguage = $("#languageSelection").val();
  const selectedOption = $("input[name='emotionRadioOptions']:checked").val();
  const styleOption = $("input[name='styleRadioOptions']:checked").val();

  const url = "/text-assistant";
  const body = {
    user_content: $("#inputTextarea").val(),
    language: selectedLanguage,
    option: selectedOption,
    style: styleOption,
  };

  const header = {
    "Content-Type": "application/json",
  };

  // get menue recommendation from API
  try {
    const { data } = await axios.post(url, body, header);
    return data;
  } catch (error) {
    console.error(`failed to get response from gpt: ${error}`);
  }
};
