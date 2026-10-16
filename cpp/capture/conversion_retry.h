#ifndef ROI_H265_CAPTURE_CONVERSION_RETRY_H_
#define ROI_H265_CAPTURE_CONVERSION_RETRY_H_

namespace roi_h265 {

struct CaptureConversionResult {
    int status;
    int attempts;
};

// Retry the conversion operation, never acquisition: recorded-source frame
// identity and capture PTS must survive a transient accelerator reset.
template <typename Convert>
CaptureConversionResult convertCaptureFrame(Convert convert, bool retry_once) {
    CaptureConversionResult result = {convert(), 1};
    if (result.status != 0 && retry_once) {
        result.status = convert();
        result.attempts = 2;
    }
    return result;
}

}  // namespace roi_h265

#endif  // ROI_H265_CAPTURE_CONVERSION_RETRY_H_
