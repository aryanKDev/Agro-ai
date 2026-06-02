// ═══════════════════════════════════════════════════════════
// AgroAI — auth.js  (Phase 1A: Multi-User Auth State Manager)
// Debugging fixes applied:
//   1. Assign Auth to window so window.Auth check works
//   2. Wrapped init() in DOMContentLoaded for safety
//   3. Added console.log debug markers at every step
//   4. Fixed _setLoading to not break button re-enabling
// ═══════════════════════════════════════════════════════════

(function () {
  'use strict';

  const API       = 'http://127.0.0.1:5000';
  const TOKEN_KEY = 'agroai_jwt';
  const USER_KEY  = 'agroai_user';

  // ── State ─────────────────────────────────────────────────
  let _token = localStorage.getItem(TOKEN_KEY) || null;
  let _user  = (function () {
    try { return JSON.parse(localStorage.getItem(USER_KEY)) || null; }
    catch (e) { return null; }
  }());

  // ── Public API ────────────────────────────────────────────
  function isLoggedIn()    { return !!_token; }
  function getToken()      { return _token; }
  function getUser()       { return _user; }
  function getAuthHeaders() {
    return _token ? { 'Authorization': 'Bearer ' + _token } : {};
  }

  function login(token, user) {
    _token = token;
    _user  = user;
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    _updateNavbar();
    _dispatchAuthChange('login');
  }

  function logout() {
    // Notify backend (fire-and-forget)
    try {
      fetch(API + '/api/auth/logout', {
        method: 'POST',
        headers: Object.assign({}, getAuthHeaders(), { 'Content-Type': 'application/json' })
      });
    } catch (_) {}

    _token = null;
    _user  = null;
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    // NOTE: agroai_history is cleared in the 'agroai:auth' event handler in script.js
    _updateNavbar();
    _dispatchAuthChange('logout');
    if (window.Router) window.Router.navigate('home');
  }

  // ── Auth API Calls ────────────────────────────────────────
  async function register(name, email, password) {
    console.log('[Auth] Sending register request for:', email);
    const res = await fetch(API + '/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, email: email, password: password })
    });
    const data = await res.json();
    console.log('[Auth] Register response:', data);
    return data;
  }

  async function loginRequest(email, password) {
    console.log('[Auth] Sending login request for:', email);
    const res = await fetch(API + '/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: email, password: password })
    });
    const data = await res.json();
    console.log('[Auth] Login response success:', data.success);
    return data;
  }

  async function fetchProfile() {
    if (!_token) return null;
    try {
      const res  = await fetch(API + '/api/auth/profile', { headers: getAuthHeaders() });
      const data = await res.json();
      return data.success ? data.user : null;
    } catch (e) { return null; }
  }

  async function updateProfile(fields) {
    const res = await fetch(API + '/api/auth/profile', {
      method: 'PUT',
      headers: Object.assign({}, getAuthHeaders(), { 'Content-Type': 'application/json' }),
      body: JSON.stringify(fields)
    });
    return await res.json();
  }

  // ── Navbar Update ─────────────────────────────────────────
  function _updateNavbar() {
    var guestLinks  = document.getElementById('nav-guest-links');
    var authLinks   = document.getElementById('nav-auth-links');
    var welcomePill = document.getElementById('nav-welcome-pill');
    var navCta      = document.getElementById('nav-cta');
    var adminBadge  = document.getElementById('btn-nav-admin');

    if (isLoggedIn() && _user) {
      if (guestLinks)  guestLinks.classList.add('hidden');
      if (authLinks)   authLinks.classList.remove('hidden');
      if (welcomePill) {
        welcomePill.textContent = '👤 ' + _user.name.split(' ')[0];
        welcomePill.classList.remove('hidden');
      }
      if (navCta) navCta.classList.add('hidden');
      // Show Admin badge only for admin role
      if (adminBadge) {
        if (_user.role === 'admin') {
          adminBadge.classList.remove('hidden');
        } else {
          adminBadge.classList.add('hidden');
        }
      }
    } else {
      if (guestLinks)  guestLinks.classList.remove('hidden');
      if (authLinks)   authLinks.classList.add('hidden');
      if (welcomePill) welcomePill.classList.add('hidden');
      if (navCta)      navCta.classList.remove('hidden');
      if (adminBadge)  adminBadge.classList.add('hidden');
    }
  }

  function _dispatchAuthChange(type) {
    document.dispatchEvent(new CustomEvent('agroai:auth', { detail: { type: type } }));
  }

  // ── Modal Controllers ─────────────────────────────────────
  function showLogin() {
    _closeAll();
    var m = document.getElementById('modal-login');
    if (m) m.classList.add('active');
  }

  function showRegister() {
    _closeAll();
    var m = document.getElementById('modal-register');
    if (m) m.classList.add('active');
  }

  function closeModals() { _closeAll(); }

  function _closeAll() {
    document.querySelectorAll('.auth-modal').forEach(function (m) {
      m.classList.remove('active');
    });
    document.querySelectorAll('.auth-error').forEach(function (e) {
      e.textContent = '';
      e.classList.add('hidden');
    });
    // Reset all form inputs so no values persist between sessions
    var fl = document.getElementById('form-login');
    var fr = document.getElementById('form-register');
    if (fl) fl.reset();
    if (fr) fr.reset();
  }

  // ── Helpers ───────────────────────────────────────────────
  function _showError(el, msg) {
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden');
    el.classList.add('auth-error-shake');
    setTimeout(function () { el.classList.remove('auth-error-shake'); }, 400);
  }

  function _setLoading(btn, on, label) {
    if (!btn) return;
    btn.disabled = on;
    if (on) {
      btn.dataset.origLabel = btn.textContent;
      btn.innerHTML = '<span class="auth-btn-spinner"></span> ' + label;
    } else {
      btn.innerHTML = label || btn.dataset.origLabel || '';
    }
  }

  // ── Init — called after DOM is ready ─────────────────────
  function init() {
    console.log('[Auth] init() called — wiring event listeners');

    _updateNavbar();

    // ── Navbar buttons ────────────────────────────────────
    var btnLogin    = document.getElementById('btn-nav-login');
    var btnRegister = document.getElementById('btn-nav-register');
    var btnLogout   = document.getElementById('btn-nav-logout');
    var btnProfile  = document.getElementById('btn-nav-profile');

    console.log('[Auth] btn-nav-login element:', btnLogin);
    console.log('[Auth] btn-nav-register element:', btnRegister);

    if (btnLogin)    btnLogin.addEventListener('click',    function (e) { e.preventDefault(); e.stopPropagation(); console.log('[Auth] Login nav clicked'); showLogin(); });
    if (btnRegister) btnRegister.addEventListener('click', function (e) { e.preventDefault(); e.stopPropagation(); console.log('[Auth] Register nav clicked'); showRegister(); });
    if (btnLogout)   btnLogout.addEventListener('click',   function (e) { e.preventDefault(); logout(); });
    if (btnProfile)  btnProfile.addEventListener('click',  function (e) { e.preventDefault(); e.stopPropagation(); if (window.Router) Router.navigate('profile'); });

    // ── Modal backdrop close ──────────────────────────────
    document.querySelectorAll('.auth-modal').forEach(function (modal) {
      modal.addEventListener('click', function (e) {
        if (e.target === modal) _closeAll();
      });
    });

    // ── Close buttons ─────────────────────────────────────
    document.querySelectorAll('.auth-modal-close').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        _closeAll();
      });
    });

    // ── Modal switch links ────────────────────────────────
    var goRegister = document.getElementById('link-go-register');
    var goLogin    = document.getElementById('link-go-login');
    if (goRegister) goRegister.addEventListener('click', function (e) { e.preventDefault(); showRegister(); });
    if (goLogin)    goLogin.addEventListener('click',    function (e) { e.preventDefault(); showLogin(); });

    // ── Register Form ─────────────────────────────────────
    var formRegister = document.getElementById('form-register');
    if (formRegister) {
      console.log('[Auth] form-register found, attaching submit listener');
      formRegister.addEventListener('submit', async function (e) {
        e.preventDefault();
        console.log('[Auth] Register form submitted');

        var btn      = document.getElementById('btn-register-submit');
        var errorEl  = document.getElementById('register-error');
        var name     = (document.getElementById('reg-name')    || {}).value || '';
        var email    = (document.getElementById('reg-email')   || {}).value || '';
        var password = (document.getElementById('reg-password')|| {}).value || '';
        var confirm  = (document.getElementById('reg-confirm') || {}).value || '';

        name  = name.trim();
        email = email.trim();

        if (password !== confirm) {
          _showError(errorEl, 'Passwords do not match.');
          return;
        }

        _setLoading(btn, true, 'Creating Account…');
        var data = await register(name, email, password);
        _setLoading(btn, false, 'Create Account');

        if (data.success) {
          _closeAll();
          if (window.Toast) Toast.show('✅ Account created! Please login.', 'success');
          showLogin();
          var loginEmailEl = document.getElementById('login-email');
          if (loginEmailEl) loginEmailEl.value = email;
        } else {
          _showError(errorEl, data.message || 'Registration failed. Please try again.');
        }
      });
    } else {
      console.warn('[Auth] form-register NOT found in DOM');
    }

    // ── Login Form ────────────────────────────────────────
    var formLogin = document.getElementById('form-login');
    if (formLogin) {
      console.log('[Auth] form-login found, attaching submit listener');
      formLogin.addEventListener('submit', async function (e) {
        e.preventDefault();
        console.log('[Auth] Login form submitted');

        var btn      = document.getElementById('btn-login-submit');
        var errorEl  = document.getElementById('login-error');
        var email    = ((document.getElementById('login-email')   || {}).value || '').trim();
        var password = ((document.getElementById('login-password')|| {}).value || '');

        _setLoading(btn, true, 'Signing In…');
        var data = await loginRequest(email, password);
        _setLoading(btn, false, 'Sign In');

        if (data.success) {
          console.log('[Auth] Login success | JWT token length:', data.token ? data.token.length : 0);
          login(data.token, data.user);
          _closeAll();
          if (window.Toast) Toast.show('🌿 Welcome back, ' + data.user.name.split(' ')[0] + '!', 'success');
          // Data re-render is handled by the 'agroai:auth' event dispatched in login()
        } else {
          _showError(errorEl, data.message || 'Login failed. Please try again.');
        }
      });
    } else {
      console.warn('[Auth] form-login NOT found in DOM');
    }

    // ── Password visibility toggles ───────────────────────
    document.querySelectorAll('.toggle-password').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var input = document.getElementById(btn.dataset.target);
        if (!input) return;
        input.type = (input.type === 'password') ? 'text' : 'password';
        btn.textContent = (input.type === 'password') ? '👁️' : '🙈';
      });
    });

    console.log('[Auth] init() complete');
  }

  // ── Expose on window ─────────────────────────────────────
  // CRITICAL: must use window.Auth = ... so window.Auth check in script.js works
  window.Auth = {
    init          : init,
    isLoggedIn    : isLoggedIn,
    getToken      : getToken,
    getUser       : getUser,
    getAuthHeaders: getAuthHeaders,
    login         : login,
    logout        : logout,
    fetchProfile  : fetchProfile,
    updateProfile : updateProfile,
    showLogin     : showLogin,
    showRegister  : showRegister,
    closeModals   : closeModals,
    register      : register,
    loginRequest  : loginRequest,
  };

  console.log('[Auth] auth.js loaded — window.Auth:', typeof window.Auth);

}());
