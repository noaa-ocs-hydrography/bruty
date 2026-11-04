-- ============================================================
-- enc_request_pause_at
-- ============================================================

CREATE OR REPLACE FUNCTION public.set_enc_request_pause_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.enc_request_pause = TRUE
       AND OLD.enc_request_pause IS DISTINCT FROM TRUE THEN

        NEW.enc_request_pause_at = clock_timestamp();
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_set_enc_request_pause_at
ON public.bruty_status;


CREATE TRIGGER trg_set_enc_request_pause_at
BEFORE UPDATE OF enc_request_pause
ON public.bruty_status
FOR EACH ROW
EXECUTE FUNCTION public.set_enc_request_pause_at();


-- ============================================================
-- request_exit_at
-- ============================================================

CREATE OR REPLACE FUNCTION public.set_request_exit_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.request_exit = TRUE
       AND OLD.request_exit IS DISTINCT FROM TRUE THEN

        NEW.request_exit_at = clock_timestamp();
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_set_request_exit_at
ON public.bruty_status;


CREATE TRIGGER trg_set_request_exit_at
BEFORE UPDATE OF request_exit
ON public.bruty_status
FOR EACH ROW
EXECUTE FUNCTION public.set_request_exit_at();

