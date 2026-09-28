const PRODUCTS = [
  {id: "blue-mug", name: "Blue ceramic mug", price: 14},
  {id: "red-mug", name: "Red ceramic mug", price: 14},
  {id: "travel-mug", name: "Steel travel mug", price: 22},
  {id: "blue-scarf", name: "Blue wool scarf", price: 35},
  {id: "green-teapot", name: "Green glass teapot", price: 41},
];
const qs = k => new URLSearchParams(location.search).get(k);
const store = {
  get: (k, d) => JSON.parse(localStorage.getItem(k) || JSON.stringify(d)),
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
};
function nav() {
  document.body.insertAdjacentHTML("afterbegin", `<nav>
    <a href="index.html">Home</a> <a href="search.html">Shop</a> <a href="articles.html">Articles</a>
    <a href="settings.html">Settings</a> <a href="login.html">Sign in</a>
    <a href="cart.html">Cart (${store.get("cart", []).length})</a></nav><hr>`);
}
