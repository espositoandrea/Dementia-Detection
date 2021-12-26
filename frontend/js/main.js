$(window).scroll(function () {
	if ($(document).scrollTop() > 50) {
		$('.navbar').removeClass('bg-transparent');
		$('.navbar').css('background-color', '#333333')
	} else {
		$('.navbar').addClass('bg-transparent');
	}
});

// Change the name of the inputfile when something is loaded

$('.inputfile').on('change', function () {
	$(this).prev().text($(this).prop('files')[0].name);
});


function formatOutput(data, format, includeBtn) {
	const printBtn = includeBtn ?
		'<a href="#" onclick="printInfo(this)" class="col-3 mb-4 btn btn-outline-primary"><i class="fa fa-print"></i> Print</a>' :
		'';
	let toReturn = data;
	switch (format) {
		case "html":
			toReturn = printBtn + '<div>' + data + '</div>';
			break;
		case "json":
			toReturn = "<div><pre>" + JSON.stringify(data, null, 4) + "</pre></div>";
			break;
		case "txt":
			toReturn = printBtn + "<div><pre>" + data + "</pre></div>";
			break;
		default:
			break;
	}
	return "<hr>" + toReturn;
}

/* attach a submit handler to the form */
$("#predict-form,#analyze-form").each(function () {
	$(this).submit(function (event) {
		event.preventDefault();

		let form = $(this);
		let formData = new FormData(form[0]);
		let url = form.attr('action');
		$.ajax({
			type: 'POST',
			url: url + "?format=" + formData.get("format"),
			data: formData,
			contentType: false,
			processData: false,
			success: (data) => {
				const toPrint = formatOutput(data, formData.get("format"), $(this).attr('id') == "analyze-form");
				$('#results').html(toPrint)
			}
		}).fail(console.log);
	});
});

function printInfo(ele) {
	console.log(ele.nextSibling)
	var openWindow = window.open("", "title", "attributes");
	openWindow.document.write(ele.nextSibling.innerHTML);
	openWindow.document.write("<style>pre {font-family: monospace;}</style>");
	openWindow.document.close();
	openWindow.focus();
	openWindow.print();
	openWindow.close();
}
