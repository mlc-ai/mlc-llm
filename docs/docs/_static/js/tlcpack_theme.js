$(document).ready(function () {
    $('#menuBtn').click(function () {
      $('#headMenu').addClass('opne');
      $('body').addClass('scroll-hide');
    });
    $('#closeHeadMenu').click(function () {
      $('#headMenu').removeClass('opne');
      $('body').removeClass('scroll-hide');
    });
    $('#button').on('click', function(e) {
      e.preventDefault();
      $('html, body').animate({scrollTop:0}, '300');
    });
  });
$(window).scroll(function () {
    var sticky = $('.wy-nav-side'),
      scroll = $(window).scrollTop();
  
    if (scroll >= 40) {
      sticky.removeClass('fixed');
    }
    else {
      sticky.addClass('fixed');
    }
});

//back to top 


$(window).scroll(function() {
  var btn = $('#button');
  if ($(window).scrollTop() >  144) {
    btn.addClass('show');
    $(".header").addClass('navigation-hide');
    $(".wy-nav-top").addClass('navigation-hide');
  } else {
    btn.removeClass('show');
    $(".header").removeClass('navigation-hide');
    $(".wy-nav-top").removeClass('navigation-hide');
  }
});

// Remove empty code blocks
document.querySelectorAll("div.highlight > pre").forEach(p => {
  if (p.innerText.trim() === "") {
      p.parentNode.removeChild(p);
  }
})
