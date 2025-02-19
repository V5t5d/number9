(function ($) {
    "use strict";

    $(document).ready(function () {

        $(document).on("click", ".smoothscroll", function (event) {
            event.preventDefault();
            var el = $(this).attr("href");
            var elWrapped = $(el);
            var header_height = $(".navbar").height();

            scrollToDiv(elWrapped, header_height);
        });

        function scrollToDiv(element, navheight) {
            if (element.length) {
                var offset = element.offset().top;
                var totalScroll = offset - navheight;

                $("html, body").animate({ scrollTop: totalScroll }, 300);
            }
        }

        $(window).on("scroll", function () {
            function isScrollIntoView(elem) {
                var docViewTop = $(window).scrollTop();
                var docViewBottom = docViewTop + $(window).height();
                var elemTop = $(elem).offset().top;
                var elemBottom = elemTop + $(window).height() * 0.5;

                if (elemBottom <= docViewBottom && elemTop >= docViewTop) {
                    $(elem).addClass("active");
                } else {
                    $(elem).removeClass("active");
                }

                var timelineContainer = $("#vertical-scrollable-timeline")[0];
                if (timelineContainer) {
                    var timelineBottom =
                        timelineContainer.getBoundingClientRect().bottom - $(window).height() * 0.5;
                    $(timelineContainer).find(".inner").css("height", timelineBottom + "px");
                }
            }

            $("#vertical-scrollable-timeline li").each(function () {
                isScrollIntoView(this);
            });
        });
    });
})(window.jQuery);
