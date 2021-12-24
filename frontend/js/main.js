$(window).scroll(function () {
  if ($(document).scrollTop() > 50) {
    $('.navbar').removeClass('bg-transparent');
    $('.navbar').css('background-color', '#333333')
  } else {
    $('.navbar').addClass('bg-transparent');
  }
});



// Change the name of the inputfile when something is loaded

$('.inputfile').on('change', function() {
  console.log($(this));
  $(this).prev().text($(this).prop('files')[0].name);
});


/* attach a submit handler to the form */
$("#predict-form").submit(function (event) {

  /* stop form from submitting normally */
  event.preventDefault();

  let xhr = new XMLHttpRequest();
  let form = $(this);
  let frame = $('#frame').prop('files')[0];
  let format = $('#predict-format').val();
  console.log(frame);
  
  let formData = new FormData();
  formData.append('image',frame);
  formData.append('format',format);
  let url = form.attr('action');
  console.log(url);
  console.log(format);
  xhr.open("POST", url);
  xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
  xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
  xhr.setRequestHeader('Content-Type', 'multipart/form-data');

  xhr.send(formData); 
  xhr.onreadystatechange = function() { 
    // If the request completed, close the extension popup
    if (xhr.readyState == 4)
      if (xhr.status == 200) {
        $('#results').text(xhr.responseText)
      }
  };
});